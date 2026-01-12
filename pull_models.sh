#!/usr/bin/env bash
# set -euo pipefail

# ====== config (edit these) ======
REMOTE_USER_HOST="acheluva3@sky2.cc.gatech.edu"
REMOTE_PATH="/coc/cedarp-dxu345-0/acheluva3/EgoVerse/logs/trained/cotrain_objcont_ft_pi_8/object in container cotrain finetune pi 8 gpu_2026-01-10_05-24-38/0/checkpoints"
LOCAL_PATH="./egomimic/robot/models/pi_cotrain_objcont"
# =================================

mkdir -p "$LOCAL_PATH"

# Prefer system rsync to avoid OpenSSL/conda mismatch
RSYNC_BIN="/usr/bin/rsync"
if [[ ! -x "$RSYNC_BIN" ]]; then
  RSYNC_BIN="$(command -v rsync)"
fi

# Run rsync without Conda/mamba library injection
env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
  "$RSYNC_BIN" -avh --progress --partial --inplace \
  --exclude='***/0/videos/***' \
  --exclude='***/0/wandb/***' \
  "${REMOTE_USER_HOST}:${REMOTE_PATH%/}/" \
  "${LOCAL_PATH%/}/"
