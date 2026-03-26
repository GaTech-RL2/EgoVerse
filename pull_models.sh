#!/usr/bin/env bash
set -euo pipefail

# ====== config (edit these) ======
REMOTE_USER_HOST="paphiwetsa3@sky1.cc.gatech.edu"
REMOTE_PATH="/coc/flash7/paphiwetsa3/projects/EgoVerse/logs/zarr/"
LOCAL_PATH="./egomimic/robot/models/s3_resolver"
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
