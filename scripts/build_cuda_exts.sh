#!/bin/bash
# Build flash-attn, causal_conv1d, mamba_ssm against the project-local
# CUDA 12.4 toolkit installed at ./cuda-12.4 (via micromamba). Run from
# the project root via:
#   srun --jobid=<JOB> --chdir=$PWD scripts/build_cuda_exts.sh
set -e
cd /coc/flash7/paphiwetsa3/projects/EgoVerse

export CUDA_HOME=$PWD/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export MAX_JOBS=4
# A40 (used by geta40_rl2) = sm_86; include sm_80 (A100) and sm_89/90
# for forward compat.

echo "=== nvcc ==="
nvcc --version | tail -2
echo "=== torch ==="
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda runtime', torch.version.cuda)"

echo "=== clear stale wheel cache (pip caches the wheel by version, not by"
echo "===  torch ABI it was built against — so old caches haunt rebuilds) ==="
.venv/bin/python -m pip cache remove "flash_attn*" 2>&1 | tail -2
.venv/bin/python -m pip cache remove "mamba_ssm*" 2>&1 | tail -2
.venv/bin/python -m pip cache remove "causal_conv1d*" 2>&1 | tail -2

# Pinned to versions known to match torch 2.6's c10 ABI.
# flash_attn  2.8+ / mamba_ssm  2.3+ use c10::Error(SourceLocation, string)
# and c10::Warning(variant<...>, ...) signatures that only exist in torch 2.7+
# and produce undefined-symbol errors against torch 2.6's libc10.so.
echo "=== installing causal_conv1d==1.4.0 (--no-deps --no-build-isolation) ==="
.venv/bin/python -m pip install --no-deps --no-build-isolation "causal_conv1d==1.4.0" 2>&1 | tail -10

echo "=== installing mamba_ssm==2.2.4 ==="
.venv/bin/python -m pip install --no-deps --no-build-isolation "mamba_ssm==2.2.4" 2>&1 | tail -10

echo "=== installing flash-attn==2.7.4.post1 ==="
.venv/bin/python -m pip install --no-deps --no-build-isolation "flash-attn==2.7.4.post1" 2>&1 | tail -10

echo "=== verifying kernel detection ==="
.venv/bin/python -c "
from egomimic.models.hnet.blocks import has_flash_attn, has_mamba
from egomimic.models.hnet.routing import has_mamba_scan
print('has_flash_attn:', has_flash_attn())
print('has_mamba:    ', has_mamba())
print('has_mamba_scan:', has_mamba_scan())
"
