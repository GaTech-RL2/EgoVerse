#!/bin/bash
set -uxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse7

if ! type module >/dev/null 2>&1; then
  source /usr/local/pace-apps/lmod/lmod/init/bash
fi
module load cuda/12.6.1
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export TORCH_CUDA_ARCH_LIST="9.0;8.9;8.0"
export MAX_JOBS=8
export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"

echo "=== reinstall mamba_ssm from main branch ==="
uv pip install --python .venv/bin/python --no-deps --no-build-isolation --reinstall \
  "git+https://github.com/state-spaces/mamba.git" 2>&1 | tail -10

echo "=== verify mamba2 import + seq_idx support ==="
.venv/bin/python <<'PY'
import torch
from mamba_ssm.modules.mamba2 import Mamba2
m = Mamba2(d_model=128, layer_idx=0).cuda().to(torch.bfloat16)
x = torch.randn(1, 16, 128, device='cuda', dtype=torch.bfloat16)
seq_idx = torch.zeros(1, 16, device='cuda', dtype=torch.int32)
seq_idx[:, 8:] = 1
out = m(x, seq_idx=seq_idx)
print("OK Mamba2 packed forward — out shape:", out.shape)
PY
