#!/bin/bash
set -e
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact

if [[ ! -x ./.micromamba/bin/micromamba ]]; then
  mkdir -p .micromamba
  curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C .micromamba bin/micromamba
fi

# cu12.4 toolkit (proven). torch is 2.7.1+cu126 but compiling against cu12.4
# nvcc works (forward-compatible runtime).
if [[ ! -x ./cuda-12.4/bin/nvcc ]]; then
  ./.micromamba/bin/micromamba create -p ./cuda-12.4 \
    -c "nvidia/label/cuda-12.4.1" \
    cuda-nvcc cuda-cudart-dev cuda-cccl cuda-nvrtc-dev cuda-libraries-dev \
    -y 2>&1 | tail -5
fi
./cuda-12.4/bin/nvcc --version | tail -2

uv pip cache remove "flash_attn*" 2>&1 | tail -1 || true
uv pip cache remove "mamba_ssm*"   2>&1 | tail -1 || true
uv pip cache remove "causal_conv1d*" 2>&1 | tail -1 || true

export CUDA_HOME=$PWD/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export MAX_JOBS=4

.venv/bin/python -c "
import torch
print(\"torch\", torch.__version__, \"/ cuda\", torch.version.cuda, \"/ avail\", torch.cuda.is_available())
"

echo "=== causal_conv1d 1.6.2.post1 ==="
uv pip install --no-deps --no-build-isolation \
  "causal_conv1d==1.6.2.post1" 2>&1 | tail -5

echo "=== mamba_ssm 2.3.2.post1 ==="
uv pip install --no-deps --no-build-isolation \
  "mamba_ssm==2.3.2.post1" 2>&1 | tail -5

echo "=== flash-attn 2.7.4.post1 (slow: 15-30 min) ==="
uv pip install --no-deps --no-build-isolation \
  "flash-attn==2.7.4.post1" 2>&1 | tail -5

.venv/bin/python -c "
import torch
print(\"torch:\", torch.__version__, \"cuda:\", torch.version.cuda)
from egomimic.models.hnet.blocks import has_flash_attn, has_mamba
from egomimic.models.hnet.routing import has_mamba_scan
print(\"flash_attn:\", has_flash_attn(), \"mamba:\", has_mamba(), \"mamba_scan:\", has_mamba_scan())
"
