#!/usr/bin/env bash
set -euo pipefail

# ====== config (edit these) ======
REMOTE_USER_HOST="paphiwetsa3@login-phoenix.pace.gatech.edu"
REMOTE_PATH="/storage/home/hcoda1/4/paphiwetsa3/r-dxu345-0/projects/EgoVerse/logs/pick_place/"
LOCAL_PATH="./egomimic/robot/models/"
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
