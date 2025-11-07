"""Training script for LLaMA model with multi-dimensional parallelism.

This script supports distributed training using four parallelism strategies:
- Tensor Parallelism (TP): Splits model weights across GPUs
- Pipeline Parallelism (PP): Splits model layers across GPUs
- Data Parallelism (DP): Replicates model across GPUs with different data
- Context Parallelism (CP): Splits sequence length across GPUs

Example usage:
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 --master_addr localhost --master_port 25500 train.py --config tmp/fast_benchmark/120M_model_tiny_stories_dp=4.json
    CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 5678 -m torch.distributed.run -- --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 train.py --config tmp/dummy/llama2_7b_benchmark.json
"""

# Standard library imports
import argparse
import datetime
import inspect
import json
import os
import time

# Local imports - parallelism and distributed training
import picotron.process_group_manager as pgm

# Third-party imports
import torch
import torch.distributed as dist
import torch.nn.functional as F
from picotron.checkpoint import (
    CheckpointManager,
    init_model_with_dematerialized_weights,
    init_model_with_materialized_weights,
)
from picotron.context_parallel.context_parallel import apply_context_parallel
from picotron.data import MicroBatchDataLoader
from picotron.data_parallel.data_parallel import DataParallelBucket

# Local imports - model, data, and utilities
from picotron.model import Llama
from picotron.pipeline_parallel.pipeline_parallel import (
    PipelineParallel,
    train_step_pipeline_1f1b,
    train_step_pipeline_afab,
)
from picotron.process_group_manager import setup_process_group_manager
from picotron.tensor_parallel.tensor_parallel import apply_tensor_parallel
from picotron.utils import (
    average_loss_across_dp_cp_ranks,
    download_model,
    get_mfu,
    get_num_params,
    print,
    set_all_seed,
    to_readable_format,
)
from torch.optim import AdamW
from transformers import AutoConfig

import wandb


def train_step(model, data_loader, device):
    """Perform a single training step with gradient accumulation.

    This function handles gradient accumulation across multiple micro-batches.
    Gradient synchronization is disabled for all but the last micro-batch to
    optimize communication overhead in distributed settings.

    Args:
        model: The model to train (may be wrapped with parallelism strategies).
        data_loader: MicroBatchDataLoader providing batches for gradient accumulation.
        device: Device to run training on (cuda or cpu).

    Returns:
        Accumulated loss across all micro-batches in this step.

    """
    acc_loss = 0.0

    # Only synchronize gradients if using data or context parallelism
    # This avoids unnecessary communication during gradient accumulation
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1

    # Process each micro-batch in the gradient accumulation sequence
    for i in range(data_loader.grad_acc_steps):
        # Get the next micro-batch from the data loader
        batch = next(data_loader)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        # Disable gradient synchronization for all but the last micro-batch
        # This allows gradients to accumulate locally before the final sync
        if requires_grad_sync:
            model.require_backward_grad_sync = i == data_loader.grad_acc_steps - 1

        # Forward pass through the model
        outputs = model(input_ids=input_ids)

        # Compute cross-entropy loss
        # Reshape outputs and targets to (seq_len * batch_size, vocab_size)
        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1)
        outputs = outputs.view(seq_len * batch_size, -1)

        # Normalize loss by gradient accumulation steps to get true average
        loss = (
            F.cross_entropy(outputs, target_ids, reduction="mean")
            / data_loader.grad_acc_steps
        )

        # Backward pass - gradients accumulate across micro-batches
        loss.backward()

        # Accumulate loss for logging (not used for optimization)
        acc_loss += loss.item()

    return acc_loss


if __name__ == "__main__":
    # ============================================================================
    # Configuration Loading
    # ============================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to config file")
    args = parser.parse_args()

    # Load training configuration from JSON file
    with open(args.config, "r") as f:
        config = json.load(f)

    # ============================================================================
    # Environment Setup
    # ============================================================================
    # Set OpenMP threads for CPU operations (affects BLAS operations)
    os.environ["OMP_NUM_THREADS"] = config["environment"]["OMP_NUM_THREADS"]
    # Control tokenizer parallelism to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = config["environment"][
        "TOKENIZERS_PARALLELISM"
    ]
    # Enable/disable Flash Attention kernel optimizations
    os.environ["FLASH_ATTEN"] = config["environment"]["FLASH_ATTEN"]
    # Set device type (cuda or cpu)
    os.environ["DEVICE"] = "cpu" if config["distributed"]["use_cpu"] else "cuda"

    # Handle HuggingFace token for model downloads
    # Priority: environment variable > config file > error
    if config["environment"].get("HF_TOKEN") is None:
        if "HF_TOKEN" not in os.environ:
            raise ValueError(
                "HF_TOKEN is neither set in the config file nor in the environment"
            )
    else:
        if "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = config["environment"]["HF_TOKEN"]
        else:
            print(
                "Warning: HF_TOKEN is set in the environment and the config file. Using the environment variable."
            )

    # Select data type: bfloat16 for GPU if supported, otherwise float32
    # bfloat16 provides better performance and memory efficiency on modern GPUs
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and not config["distributed"]["use_cpu"]
        else torch.float32
    )
    # Flash Attention requires bfloat16 when enabled
    assert (dtype == torch.bfloat16 and os.getenv("FLASH_ATTEN") == "1") or os.getenv(
        "FLASH_ATTEN"
    ) != "1", "Kernel operations requires dtype=torch.bfloat16"

    # ============================================================================
    # Distributed Training Setup
    # ============================================================================
    # Extract rank information from environment (set by torchrun)
    local_rank = int(os.environ["LOCAL_RANK"])  # Rank within current node
    global_rank = int(os.environ["RANK"])  # Rank across all nodes
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes

    # Select communication backend: NCCL for GPU, Gloo for CPU
    backend = "gloo" if config["distributed"]["use_cpu"] else "nccl"

    # Validate configuration constraints
    # Context parallelism requires sequence length to be divisible by cp_size
    assert (
        config["training"]["seq_length"] % config["distributed"]["cp_size"] == 0
    ), "seq_length must be divisible by cp_size for Context Parallelism"
    # Total world size must match the product of all parallelism dimensions
    assert (
        world_size
        == config["distributed"]["tp_size"]
        * config["distributed"]["pp_size"]
        * config["distributed"]["dp_size"]
        * config["distributed"]["cp_size"]
    ), "world_size must be equal to tp_size * pp_size * dp_size * cp_size"

    # Set device for this process
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # Initialize PyTorch distributed process group
    # This enables communication between all processes
    dist.init_process_group(
        rank=global_rank,
        world_size=world_size,
        backend=backend,
        init_method=f"env://",
        timeout=datetime.timedelta(minutes=3),
    )

    # Setup process group manager for multi-dimensional parallelism
    # Creates communication groups for TP, CP, PP, and DP
    setup_process_group_manager(
        tp_size=config["distributed"]["tp_size"],
        cp_size=config["distributed"]["cp_size"],
        pp_size=config["distributed"]["pp_size"],
        dp_size=config["distributed"]["dp_size"],
    )

    # Determine which rank should log to wandb
    # Only one rank per parallel group logs to avoid duplicate entries
    # Requirements: first TP rank, first DP rank, first CP rank, last PP stage
    is_wandb_rank = (
        pgm.process_group_manager.tp_rank == 0
        and pgm.process_group_manager.dp_rank == 0
        and pgm.process_group_manager.cp_rank == 0
        and pgm.process_group_manager.pp_is_last_stage
    )

    # Set random seed for reproducibility across all processes
    set_all_seed(config["training"]["seed"])

    # ============================================================================
    # Data Loading Setup
    # ============================================================================
    start_time = time.time()

    # Initialize data loader with micro-batch support and gradient accumulation
    # The loader handles tokenization, sequence chunking, and distributed sampling
    data_loader = MicroBatchDataLoader(
        micro_batch_size=config["training"]["micro_batch_size"],
        seq_length=config["training"]["seq_length"],
        dataset_name=config["dataset"]["name"],
        tokenizer_name=config["model"]["name"],
        grad_acc_steps=config["training"]["gradient_accumulation_steps"],
        device=device,
        num_workers=config["dataset"]["num_workers"],
        num_proc=config["dataset"]["num_proc"],
        num_samples=config["training"].get("num_samples", None),
        subset_name=config["dataset"].get("subset_name", None),
        split=config["dataset"].get("split", "train"),
    )

    # Download model weights on rank 0 only
    # Assumes all ranks share the same filesystem (common in cluster setups)
    if pgm.process_group_manager.global_rank == 0:
        download_model(config["model"]["name"], os.environ["HF_TOKEN"])

    # Wait for all processes to reach this point before continuing
    dist.barrier()

    print(
        f"init dataloader time: {time.time()-start_time:.2f}s",
        is_print_rank=is_wandb_rank,
    )

    # Calculate total tokens processed per training step
    # This includes all micro-batches and all data parallel replicas
    tokens_per_step = data_loader.global_batch_size * config["training"]["seq_length"]

    if pgm.process_group_manager.global_rank == 0:
        print(
            "Tokens per step:",
            to_readable_format(tokens_per_step),
            is_print_rank=is_wandb_rank,
        )

    # Initialize Weights & Biases logging (only on designated rank)
    if is_wandb_rank and config["logging"]["use_wandb"]:
        wandb.init(
            project="picotron",
            name=f"{config['logging']['run_name']}_{to_readable_format(tokens_per_step)}_{pgm.process_group_manager}",
            config={
                "tensor_parallel_size": pgm.process_group_manager.tp_world_size,
                "context_parallel_size": pgm.process_group_manager.cp_world_size,
                "pipeline_parallel_size": pgm.process_group_manager.pp_world_size,
                "data_parallel_size": pgm.process_group_manager.dp_world_size,
                "model": config["model"]["name"],
                "dataset": config["dataset"]["name"],
                "max_tokens": config["training"]["max_tokens"],
                "learning_rate": config["training"]["learning_rate"],
                "seed": config["training"]["seed"],
                "micro_batch_size": data_loader.micro_batch_size,
                "global_batch_size": data_loader.global_batch_size,
                "gradient_accumulation": data_loader.grad_acc_steps,
            },
        )

    # ============================================================================
    # Model Configuration
    # ============================================================================
    # Create model configuration on rank 0, then broadcast to all ranks
    if pgm.process_group_manager.global_rank == 0:
        print(f"rank {pgm.process_group_manager.global_rank}: Creating model config")
        # Load base configuration from HuggingFace model
        model_config = AutoConfig.from_pretrained(config["model"]["name"])

        # Override model architecture parameters if specified in config
        # This allows training smaller variants of large models
        model_config.num_hidden_layers = (
            model_config.num_hidden_layers
            if "num_hidden_layers" not in config["model"]
            else config["model"]["num_hidden_layers"]
        )
        model_config.num_attention_heads = (
            model_config.num_attention_heads
            if "num_attention_heads" not in config["model"]
            else config["model"]["num_attention_heads"]
        )
        model_config.num_key_value_heads = (
            model_config.num_key_value_heads
            if "num_key_value_heads" not in config["model"]
            else config["model"]["num_key_value_heads"]
        )
        # Set maximum sequence length to match training configuration
        model_config.max_position_embeddings = config["training"]["seq_length"]
        objects = [model_config]
    else:
        objects = [None]

    # Broadcast model configuration from rank 0 to all processes
    # Ensures all ranks have identical model architecture
    dist.broadcast_object_list(objects, src=0, device=device)
    model_config = objects[0]
    print(
        f"rank {pgm.process_group_manager.global_rank}: Broadcasting model_config to all ranks",
        is_print_rank=pgm.process_group_manager.global_rank == 0,
    )

    dist.barrier()

    # ============================================================================
    # Model Initialization with Multi-Dimensional Parallelism
    # ============================================================================
    print(
        f"rank {pgm.process_group_manager.global_rank}: Initializing model meta device",
        is_print_rank=is_wandb_rank,
    )

    start_time = time.time()

    # Initialize model with dematerialized weights to save memory
    # Weights are created on meta device (no actual memory allocation)
    # This is crucial for large models that don't fit in memory
    with init_model_with_dematerialized_weights():
        # Create base LLaMA model architecture
        model = Llama(config=model_config)

        # Apply Tensor Parallelism (TP) - splits weights across TP dimension
        # Must be applied before pipeline parallelism to ensure correct layer distribution
        if pgm.process_group_manager.tp_world_size > 1:
            model = apply_tensor_parallel(model)

        # Apply Pipeline Parallelism (PP) - splits layers across PP dimension
        # Applied after TP so each pipeline stage can have its own TP group
        if pgm.process_group_manager.pp_world_size > 1:
            model = PipelineParallel(model, model_config)

    # Materialize weights from HuggingFace safetensors format
    # Loads actual weight values into memory, distributed according to TP/PP
    model = init_model_with_materialized_weights(
        model, model_config, save_dir=f"./hf_model_safetensors/"
    )

    # TODO: load existing checkpoint here to continue pre-training

    # Apply Context Parallelism (CP) - splits sequence length across CP dimension
    # Applied after weight materialization as it modifies attention computation
    if pgm.process_group_manager.cp_world_size > 1:
        model = apply_context_parallel(model)

    # Move model to target device and dtype
    model.to(dtype).to(device)

    # Apply Data Parallelism (DP) - replicates model across DP dimension
    # Applied last as it wraps the entire model for gradient synchronization
    if pgm.process_group_manager.dp_world_size > 1:
        model = DataParallelBucket(model)

    # Note: Parallelism application order is critical:
    # TP → PP → CP → DP ensures correct weight distribution and communication

    print(
        f"init model parallel time: {time.time()-start_time:.2f}s",
        is_print_rank=is_wandb_rank,
    )

    # Set model to training mode (enables dropout, etc.)
    model.train()

    # Count and report total model parameters
    num_params = get_num_params(model)
    print(
        f"Number of parameters: {to_readable_format(num_params)}",
        is_print_rank=is_wandb_rank,
    )

    # Define tensor shapes for pipeline parallelism
    # Used by pipeline engines to pre-allocate communication buffers
    tensor_shapes = (
        data_loader.micro_batch_size,
        data_loader.seq_length_per_gpu,
        model_config.hidden_size,
    )

    # ============================================================================
    # Optimizer and Checkpointing Setup
    # ============================================================================
    # Configure optimizer with optional fused AdamW for better performance
    extra_args = dict()
    if config["model"]["use_fused_adam"]:
        # Check if fused AdamW is available (PyTorch 2.0+)
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # Fused optimizer only works on CUDA
        use_fused = fused_available and device == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()

    # Initialize AdamW optimizer with learning rate from config
    optimizer = AdamW(
        model.parameters(), lr=config["training"]["learning_rate"], **extra_args
    )

    # Initialize checkpoint manager for saving/loading training state
    checkpoint_manager = CheckpointManager()

    # Initialize training state
    trained_tokens, step = 0, 0

    # Load checkpoint if specified to resume training
    if config["checkpoint"]["load_path"]:
        step, trained_tokens = checkpoint_manager.load_checkpoint(
            model, optimizer, config["checkpoint"]["load_path"]
        )

    dist.barrier()

    # ============================================================================
    # Main Training Loop
    # ============================================================================
    # Continue training until max_tokens or total_train_steps is reached
    while (
        config["training"]["max_tokens"] is None
        or trained_tokens < config["training"]["max_tokens"]
    ):
        step_start_time = time.time()

        # Zero gradients before forward pass
        optimizer.zero_grad()

        # Execute training step based on parallelism configuration
        if pgm.process_group_manager.pp_world_size > 1:
            # Pipeline parallelism: use specialized pipeline training functions
            # AFAB: All-Forward-All-Backward (simpler but less efficient)
            # 1F1B: One-Forward-One-Backward (more efficient, better GPU utilization)
            if config["distributed"]["pp_engine"] == "afab":
                loss = train_step_pipeline_afab(
                    model, data_loader, tensor_shapes, device, dtype
                )
            elif config["distributed"]["pp_engine"] == "1f1b":
                loss = train_step_pipeline_1f1b(
                    model, data_loader, tensor_shapes, device, dtype
                )
            else:
                raise ValueError(
                    f"Invalid pipeline parallel engine: {config['distributed']['pp_engine']}"
                )
        else:
            # No pipeline parallelism: use standard training step
            loss = train_step(model, data_loader, device)

        # Average loss across data parallel and context parallel ranks
        # Ensures consistent loss values across replicas
        loss = average_loss_across_dp_cp_ranks(loss, device)

        # Update model parameters using accumulated gradients
        optimizer.step()

        # Update training progress tracking
        trained_tokens += tokens_per_step
        step += 1

        # Reset model state if needed (e.g., for pipeline parallelism)
        if hasattr(model, "reset"):
            model.reset()

        # Calculate performance metrics
        step_duration = time.time() - step_start_time
        tokens_per_second = tokens_per_step / step_duration
        tokens_per_second_per_gpu = tokens_per_second / world_size
        # MFU: Model FLOPs Utilization - measures how efficiently we use compute
        mfu = get_mfu(tokens_per_second_per_gpu, num_params, model_config)

        # Log training progress (only on designated wandb rank)
        if is_wandb_rank:
            # Print formatted training metrics to console
            print(
                f"[rank {pgm.process_group_manager.global_rank}] "
                f"Step: {step:<5d} | "
                f"Loss: {loss:6.4f} | "
                f"Global batch size: {to_readable_format(tokens_per_step):>7s} | "
                f"Tokens/s: {to_readable_format(tokens_per_second):>7s} | "
                f"Tokens/s/GPU: {to_readable_format(tokens_per_second_per_gpu):>7s} | "
                f"Tokens: {to_readable_format(trained_tokens):>7s}{('/' + to_readable_format(config['training']['max_tokens'])) if config['training']['max_tokens'] else ''} | "
                f"MFU: {mfu:5.2f}% | "
                f"Memory usage: {torch.cuda.memory_reserved() / 1e9:6.2f}GB",
                is_print_rank=is_wandb_rank,
            )

            # Log metrics to Weights & Biases for experiment tracking
            if config["logging"]["use_wandb"]:
                wandb.log(
                    {
                        "loss": loss,
                        "tokens_per_step": tokens_per_step,
                        "tokens_per_second": tokens_per_step / step_duration,
                        "mfu": mfu,
                        "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                        "memory_usage": torch.cuda.memory_reserved() / 1e9,
                        "trained_tokens": trained_tokens,
                    }
                )

        # Save checkpoint at specified frequency
        # Checkpoints include model weights, optimizer state, and training progress
        if step % config["checkpoint"]["save_frequency"] == 0:
            checkpoint_manager.save_checkpoint(
                model,
                optimizer,
                step,
                trained_tokens,
                config["checkpoint"]["save_dir"] + f"/{step}",
            )

        # Exit training loop if total step limit is reached
        if step >= config["training"]["total_train_steps"]:
            break

    # ============================================================================
    # Cleanup
    # ============================================================================
    # Finalize wandb logging session
    if is_wandb_rank and config["logging"]["use_wandb"]:
        wandb.finish()

    # Clean up distributed process group
    # Releases communication resources and closes connections
    dist.destroy_process_group()
