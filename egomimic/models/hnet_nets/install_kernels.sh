#!/usr/bin/env bash
#
# Optional kernel deps for egomimic/models/hnet_nets.
#
# The H-Net code runs without these (pure-PyTorch fallbacks), but installing
# them unlocks:
#   - flash_attn       -> packed-mode attention via flash_attn_varlen_func
#                         (otherwise SDPA + block-diagonal mask)
#   - causal-conv1d    -> required by mamba_ssm
#   - mamba-ssm        -> 'm'/'M' arch blocks (Mamba2 mixer) AND parallel
#                         selective-scan kernel inside DeChunkLayer
#                         (otherwise pure-Python EMA recurrence)
#   - einops           -> tensor rearrange used by the mamba scan path
#
# Requirements:
#   - CUDA-capable GPU + matching CUDA toolkit
#   - PyTorch already installed (these wheels are torch-ABI specific)
#   - ninja (for fused build)
#
# Usage:
#   bash egomimic/models/hnet_nets/install_kernels.sh
#
# Pinned versions reflect the upstream H-Net repo's known-good combo. Bump
# them only if you've validated against your CUDA / PyTorch combo.

set -euo pipefail

# ---- sanity check: torch + CUDA ----
python - <<'PY'
import sys, torch
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("WARNING: torch.cuda.is_available() is False. These kernels are CUDA-only.", file=sys.stderr)
PY

# ---- 1) build helpers ----
pip install --upgrade pip setuptools wheel
pip install ninja packaging einops

# ---- 2) causal-conv1d (mamba_ssm dep) ----
pip install --no-build-isolation "causal-conv1d>=1.4.0"

# ---- 3) mamba-ssm (selective-scan kernel + Mamba2 block) ----
pip install --no-build-isolation "mamba-ssm>=2.2.2"

# ---- 4) flash-attn (varlen packed attention) ----
# --no-build-isolation matches flash-attn's documented install path.
pip install --no-build-isolation "flash-attn>=2.6.0"

# ---- 5) verify ----
python - <<'PY'
def check(name, expr):
    try:
        exec(expr, {})
        print(f"  ok   {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")

print("Verifying imports:")
check("einops",             "import einops")
check("causal_conv1d",      "from causal_conv1d import causal_conv1d_fn")
check("mamba_ssm.Mamba2",   "from mamba_ssm.modules.mamba2 import Mamba2")
check("mamba_chunk_scan",   "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined")
check("flash_attn_varlen",  "from flash_attn import flash_attn_varlen_func")

import torch
from egomimic.models.hnet_nets.blocks  import has_flash_attn, has_mamba
from egomimic.models.hnet_nets.routing import has_mamba_scan
print(f"\nRuntime detection:")
print(f"  has_flash_attn  = {has_flash_attn()}")
print(f"  has_mamba       = {has_mamba()}")
print(f"  has_mamba_scan  = {has_mamba_scan()}")
PY
