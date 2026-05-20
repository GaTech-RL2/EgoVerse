#!/bin/bash
# Recovery script: mamba_ssm pulled torch 2.12+cu130 over our 2.6+cu124.
# This restores torch and then rebuilds the CUDA exts with --no-deps so
# pip can't touch torch again. Run from project root via:
#   srun --jobid=<JOB> --chdir=$PWD scripts/recover_torch_cu124.sh
set -e
cd /coc/flash7/paphiwetsa3/projects/EgoVerse

export CUDA_HOME=$PWD/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export MAX_JOBS=4

echo "=== uninstall wrongly-built exts ==="
.venv/bin/python -m pip uninstall -y causal_conv1d mamba_ssm 2>&1 | tail -5

echo "=== restore torch 2.6.0+cu124 / torchvision 0.21.0+cu124 ==="
.venv/bin/python -m pip install --no-deps --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 2>&1 | tail -10

echo "=== verify torch ==="
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" 2>&1 | tail -3

echo "=== install causal_conv1d (--no-deps so pip cannot bump torch) ==="
.venv/bin/python -m pip install --no-deps --no-build-isolation causal_conv1d 2>&1 | tail -10

echo "=== install mamba_ssm --no-deps ==="
.venv/bin/python -m pip install --no-deps --no-build-isolation mamba_ssm 2>&1 | tail -10

echo "=== install flash-attn --no-deps ==="
.venv/bin/python -m pip install --no-deps --no-build-isolation flash-attn 2>&1 | tail -10

echo "=== verify ==="
.venv/bin/python -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
from egomimic.models.hnet_nets.blocks import has_flash_attn, has_mamba
from egomimic.models.hnet_nets.routing import has_mamba_scan
print('has_flash_attn:', has_flash_attn())
print('has_mamba:    ', has_mamba())
print('has_mamba_scan:', has_mamba_scan())
"
