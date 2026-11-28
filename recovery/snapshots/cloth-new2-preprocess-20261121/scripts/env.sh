#!/usr/bin/env bash
set -euo pipefail
export CAMPAIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$CAMPAIGN_ROOT" in
 /coc/flash7/cxin34/*) export CAMPAIGN_PYTHON=/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python ;;
 /storage/project/r-dxu345-0/cxin34/*) export CAMPAIGN_PYTHON="$CAMPAIGN_ROOT/env/bin/python" ;;
 *) echo 'Unsupported deployment root' >&2; return 2 ;;
esac
export PYTHONPATH="$CAMPAIGN_ROOT:$CAMPAIGN_ROOT/source"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline OMP_NUM_THREADS=4
export HF_HOME="$CAMPAIGN_ROOT/cache/huggingface" TORCH_HOME="$CAMPAIGN_ROOT/cache/torch"
export TMPDIR="/tmp/cxin34_pretraining_${SLURM_JOB_ID:-check}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME"
