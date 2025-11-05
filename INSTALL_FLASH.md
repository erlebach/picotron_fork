```bash
uv venv
source .venv/bin/activate
uv pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install setuptools wheel ninja packaging psutil
uv pip install flash-attn==2.5.9.post1 --no-build-isolation --reinstall --no-cache
```
