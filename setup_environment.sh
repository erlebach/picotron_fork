export LD_LIBRARY_PATH=/gpfs/research/scratch/gerlebacher/localpython/py311/lib:$LD_LIBRARY_PATH
module load webproxy
module load python-uv
module load cuda/12.1
uv sync
uv pip install flash-attn==2.5.9.post1 --no-deps --no-build-isolation --reinstall --no-cache
python -c "import flash_attn, flash_attn_2_cuda; print('FlashAttention OK')"
source .venv/bin/activate
# Works with cuda 12.1 and torch 2.3.1
# uv pip install flash-attn==2.5.9.post1 --no-deps --no-build-isolation --reinstall --no-cache
