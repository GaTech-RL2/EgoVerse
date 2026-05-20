#!/bin/bash
# install_cuda_kernels_pace.sh — one-shot installer for the optional H-Net
# CUDA kernels (causal_conv1d / mamba_ssm / flash_attn) on PACE.
#
# Differs from skynet version because:
#   - PACE has a system lmod-managed CUDA 12.6.1 toolkit (no need for
#     project-local micromamba + cuda-12.4).
#   - PACE venv is uv-managed (no pip binary in .venv/bin); installs go
#     through `uv pip install --python .venv/bin/python ...`.
#   - PACE has torch 2.7.1+cu126 (newer than sky1's 2.6.0+cu124), so the
#     kernel versions are different: causal_conv1d 1.5.0.post8 still OK,
#     mamba_ssm 2.2.5+ (2.2.4 was sky1-pinned for torch 2.6 ABI),
#     flash-attn 2.7.4.post1 still matches.
#
# Run from project root in an interactive SLURM allocation with a GPU:
#   srun --jobid=<JOB> --ntasks=1 --overlap scripts/install_cuda_kernels_pace.sh
# or as a wrapped one-shot:
#   nohup srun --jobid=<JOB> --ntasks=1 --overlap bash -lc \
#     'cd <PROJ> && scripts/install_cuda_kernels_pace.sh' > install.log 2>&1 < /dev/null &
#
# Requires: an active interactive SLURM allocation. Login node has no GPU
# and the CUDA-arch detection will fail.
set -euxo pipefail

# Allow caller to override; defaults assume CWD is project root.
PROJ="${PROJ:-$PWD}"
cd "$PROJ"

# ----- 1. activate venv + ensure uv on PATH -----
source .venv/bin/activate
export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"

# ----- 2. load CUDA toolkit module (matches torch 2.7.1's cu126 ABI) -----
# lmod's `module` is a bash function; make sure we're in a login-ish shell.
# If `module` isn't defined, source it manually.
if ! type module >/dev/null 2>&1; then
  source /usr/local/pace-apps/lmod/lmod/init/bash
fi
module load cuda/12.6.1

# Derive CUDA_HOME from nvcc location (works with any cuda module version).
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"

# H200 = sm_90; include older arches if you want one build to work everywhere.
# (Default to H200-only; bump if running on A100/L40S nodes too.)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;8.9;8.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "=== nvcc ==="
nvcc --version | tail -2
echo "=== torch ==="
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda runtime', torch.version.cuda)"

# ----- 3. install pinned kernel versions -----
# Use --no-deps to stop uv/pip from upgrading torch mid-install.
# Use --no-build-isolation so the build sees the active venv's torch.
# Install from the source tag on GitHub — the pypi sdist for 1.5.0.post8
# ships without csrc/causal_conv1d.cpp on some platforms (torch 2.7 + uv on PACE),
# so the build fails with "fatal error: csrc/causal_conv1d.cpp: No such file".
# The github tarball has the file in place.
echo "=== installing causal_conv1d (git tag v1.5.0.post8) ==="
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
  "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.0.post8" 2>&1 | tail -10

echo "=== installing mamba_ssm (git tag v2.2.5) ==="
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
  "git+https://github.com/state-spaces/mamba.git@v2.2.5" 2>&1 | tail -10

if [[ "${SKIP_FLASH:-0}" == "1" ]]; then
  echo "=== skipping flash-attn (SKIP_FLASH=1) — mamba is what we need for now ==="
else
  echo "=== installing flash-attn==2.7.4.post1 (slow: 15-30 min) ==="
  uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
    "flash-attn==2.7.4.post1" 2>&1 | tail -10
fi

# ----- 4. verify -----
echo "=== verifying kernel detection ==="
.venv/bin/python -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
from egomimic.models.hnet_nets.blocks import has_flash_attn, has_mamba
from egomimic.models.hnet_nets.routing import has_mamba_scan
fa, m, ms = has_flash_attn(), has_mamba(), has_mamba_scan()
print(f'  has_flash_attn : {fa}')
print(f'  has_mamba      : {m}')
print(f'  has_mamba_scan : {ms}')
assert m and ms, 'mamba kernels failed to load — check above output'
print('OK (flash_attn optional)')
"
