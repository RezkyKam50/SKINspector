#!/bin/bash

# to avoid dependency conflict, build flash-attention2 first before anything else.
# is best to read flash-attention docs first to determine your GPU architecture compatibility.
uv pip install wheel flash-attn --no-build-isolation
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

chmod +x scripts/cuda_python_binding.sh && ./scripts/cuda_python_binding.sh
uv pip install -r requirements.txt