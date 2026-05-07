#!/usr/bin/env bash
set -euo pipefail

# Usage: ./pull_run.sh <run_name>
# Downloads the dataset and checkpoint for a run that shares the same name.
# Example: ./pull_run.sh RBY1_human_data_0417_pickupbox

# ====== config (edit these) ======
REMOTE_USER_HOST="sky2-zhenyang"
REMOTE_BASE="/coc/flash7/zhenyang/EgoVerse"
LOCAL_DATASET_DIR="./datasets"
LOCAL_CHECKPOINT_DIR="./checkpoints"
# =================================

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <run_name>" >&2
  exit 1
fi

NAME="$1"

mkdir -p "${LOCAL_DATASET_DIR}/${NAME}"
mkdir -p "${LOCAL_CHECKPOINT_DIR}/${NAME}"

# Prefer system rsync to avoid OpenSSL/conda mismatch
RSYNC_BIN="/usr/bin/rsync"
if [[ ! -x "$RSYNC_BIN" ]]; then
  RSYNC_BIN="$(command -v rsync)"
fi

echo "==> Pulling dataset: ${NAME}"
if ! env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
  "$RSYNC_BIN" -avh --progress --partial --inplace \
  "${REMOTE_USER_HOST}:${REMOTE_BASE}/datasets/${NAME}_human_data/" \
  "${LOCAL_DATASET_DIR}/${NAME}/"; then
  echo "Warning: dataset pull failed for ${NAME}; continuing to checkpoint pull." >&2
fi

echo "==> Pulling checkpoint: ${NAME}"
env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
  "$RSYNC_BIN" -avh --progress --partial --inplace \
  --exclude='*/videos/' \
  --exclude='*/wandb/' \
  "${REMOTE_USER_HOST}:${REMOTE_BASE}/logs/${NAME}/" \
  "${LOCAL_CHECKPOINT_DIR}/${NAME}/"

echo "==> Done."
