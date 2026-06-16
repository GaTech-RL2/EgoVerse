#!/bin/bash
set -uxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse7

# load cuda module
if ! type module >/dev/null 2>&1; then
  source /usr/local/pace-apps/lmod/lmod/init/bash
fi
module load cuda/12.6.1
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export TORCH_CUDA_ARCH_LIST="9.0;8.9;8.0"
export MAX_JOBS=8
export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"

echo "=== reinstall causal_conv1d from main branch ==="
uv pip install --python .venv/bin/python --no-deps --no-build-isolation --reinstall \
  "git+https://github.com/Dao-AILab/causal-conv1d.git" 2>&1 | tail -10

echo "=== verify cpp_functions module exists ==="
.venv/bin/python <<'PY'
import causal_conv1d
print("causal_conv1d.__file__:", causal_conv1d.__file__)
print("submodules:", [x for x in dir(causal_conv1d) if not x.startswith("_")])
from causal_conv1d.cpp_functions import causal_conv1d_fwd_function, causal_conv1d_bwd_function, causal_conv1d_update_function
print("cpp_functions OK:")
print("  fwd:", causal_conv1d_fwd_function)
print("  bwd:", causal_conv1d_bwd_function)
print("  upd:", causal_conv1d_update_function)
PY
