#!/bin/bash
# install_cuda_kernels.sh — one-shot installer for the optional H-Net
# CUDA kernels (flash_attn / mamba_ssm / causal_conv1d) against torch
# 2.6.0+cu124.
#
# What it does:
#   1. Drops micromamba into ./.micromamba/bin (if not already there).
#   2. Creates a project-local cu12.4 toolkit at ./cuda-12.4 (nvcc +
#      headers + dev libs) using the nvidia/label/cuda-12.4.1 conda
#      channel.
#   3. Clears stale pip wheel caches for the three exts (the cache
#      key is version, not torch ABI — old caches haunt rebuilds).
#   4. Builds + installs the three exts with `--no-deps
#      --no-build-isolation` (both flags REQUIRED — --no-deps stops pip
#      from upgrading torch to 2.12+cu130 mid-install).
#   5. Verifies has_flash_attn / has_mamba / has_mamba_scan all True.
#
# Run from project root via:
#   srun --jobid=<JOB> --chdir=$PWD scripts/install_cuda_kernels.sh
#
# Requires: an interactive SLURM allocation with a GPU. Login-node
# sky1 has no GPU and the build's CUDA-arch detection will fail.
set -e
cd /coc/flash7/paphiwetsa3/projects/EgoVerse

# ----- 1. micromamba -----
if [[ ! -x ./.micromamba/bin/micromamba ]]; then
  echo "=== fetching micromamba ==="
  mkdir -p .micromamba
  curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C .micromamba bin/micromamba
fi
./.micromamba/bin/micromamba --version

# ----- 2. project-local cu12.4 toolkit -----
if [[ ! -x ./cuda-12.4/bin/nvcc ]]; then
  echo "=== installing cu12.4 toolkit under ./cuda-12.4 (~1.6 GB) ==="
  ./.micromamba/bin/micromamba create -p ./cuda-12.4 \
    -c "nvidia/label/cuda-12.4.1" \
    cuda-nvcc cuda-cudart-dev cuda-cccl cuda-nvrtc-dev cuda-libraries-dev \
    -y
fi
./cuda-12.4/bin/nvcc --version | tail -2

# ----- 3. clear stale wheel caches -----
echo "=== clearing stale pip wheel caches for the three exts ==="
.venv/bin/python -m pip cache remove "flash_attn*" 2>&1 | tail -2 || true
.venv/bin/python -m pip cache remove "mamba_ssm*" 2>&1 | tail -2 || true
.venv/bin/python -m pip cache remove "causal_conv1d*" 2>&1 | tail -2 || true

# ----- 4. build environment + the three exts -----
export CUDA_HOME=$PWD/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # A100, A40, L40, H100
export MAX_JOBS=4                              # cap parallel nvcc to avoid OOM

echo "=== torch sanity ==="
.venv/bin/python -c "
import torch
assert torch.version.cuda == '12.4', f'torch built for cuda {torch.version.cuda}, expected 12.4'
assert torch.__version__.startswith('2.6.'), f'torch {torch.__version__}, expected 2.6.x'
print('torch', torch.__version__, '/ cuda', torch.version.cuda, '/ cuda_avail', torch.cuda.is_available())
"

# Pinned versions matching torch 2.6's c10 ABI:
#  - flash-attn 2.8+      / mamba_ssm 2.3+ use c10::Error(SourceLocation, std::string) and
#    c10::Warning(variant<...>, ...) signatures that exist only in torch 2.7+ libc10.so.
#  - causal_conv1d 1.4.0  has a different csrc/ layout that breaks the
#    setup.py ninja build (csrc/causal_conv1d.cpp missing).
echo "=== building causal_conv1d==1.5.0.post8 ==="
.venv/bin/python -m pip install --no-deps --no-build-isolation \
  "causal_conv1d==1.5.0.post8" 2>&1 | tail -8

echo "=== building mamba_ssm==2.2.4 ==="
.venv/bin/python -m pip install --no-deps --no-build-isolation \
  "mamba_ssm==2.2.4" 2>&1 | tail -8

echo "=== building flash-attn==2.7.4.post1 (slow: ~15-30 min on A40) ==="
.venv/bin/python -m pip install --no-deps --no-build-isolation \
  "flash-attn==2.7.4.post1" 2>&1 | tail -8

# ----- 5. verify -----
echo "=== verifying kernel detection ==="
.venv/bin/python -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
from egomimic.models.hnet.blocks import has_flash_attn, has_mamba
from egomimic.models.hnet.routing import has_mamba_scan
fa, m, ms = has_flash_attn(), has_mamba(), has_mamba_scan()
print(f'  has_flash_attn : {fa}')
print(f'  has_mamba      : {m}')
print(f'  has_mamba_scan : {ms}')
assert fa and m and ms, 'one or more kernels failed to load — check above output'
print('OK')
"
