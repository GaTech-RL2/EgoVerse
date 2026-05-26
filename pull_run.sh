#!/usr/bin/env bash
set -euo pipefail

# Usage: ./pull_run.sh <run_name> [epoch]
# Downloads the dataset and checkpoint for a run that shares the same name.
# If epoch is set, only checkpoint files matching that epoch are pulled.
# Example: ./pull_run.sh RBY1_human_data_0417_pickupbox
# Example: ./pull_run.sh RBY1_human_data_0417_pickupbox 599

# ====== config (edit these) ======
REMOTE_USER_HOST="sky2-zhenyang"
REMOTE_BASE="/coc/flash7/zhenyang/EgoVerse"
LOCAL_DATASET_DIR="./datasets"
LOCAL_CHECKPOINT_DIR="./checkpoints"
# =================================

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <run_name> [epoch]" >&2
  exit 1
fi

NAME="$1"
EPOCH="${2:-}"

if [[ -n "$EPOCH" && ! "$EPOCH" =~ ^[0-9]+$ ]]; then
  echo "Error: epoch must be a non-negative integer, got: ${EPOCH}" >&2
  exit 1
fi

mkdir -p "${LOCAL_CHECKPOINT_DIR}/${NAME}"

# Prefer system rsync to avoid OpenSSL/conda mismatch
RSYNC_BIN="/usr/bin/rsync"
if [[ ! -x "$RSYNC_BIN" ]]; then
  RSYNC_BIN="$(command -v rsync)"
fi

if [[ -d "${LOCAL_DATASET_DIR}/${NAME}" && -n "$(find "${LOCAL_DATASET_DIR}/${NAME}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "==> Dataset exists, skipping: ${LOCAL_DATASET_DIR}/${NAME}"
else
  mkdir -p "${LOCAL_DATASET_DIR}/${NAME}"
  echo "==> Pulling dataset: ${NAME}"
  if ! env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
    "$RSYNC_BIN" -avh --progress --partial --inplace \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/datasets/${NAME}_human_data/" \
    "${LOCAL_DATASET_DIR}/${NAME}/"; then
    echo "Warning: dataset pull failed for ${NAME}; continuing to checkpoint pull." >&2
  fi
fi

if [[ -n "$EPOCH" ]]; then
  echo "==> Pulling checkpoint: ${NAME}, epoch ${EPOCH}"
  env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
    "$RSYNC_BIN" -avh --progress --partial --inplace \
    --include='*/' \
    --include="*epoch=${EPOCH}.ckpt" \
    --exclude='*' \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/logs/${NAME}/" \
    "${LOCAL_CHECKPOINT_DIR}/${NAME}/"
else
  echo "==> Pulling checkpoint: ${NAME}"
  env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
    "$RSYNC_BIN" -avh --progress --partial --inplace \
    --exclude='*/videos/' \
    --exclude='*/wandb/' \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/logs/${NAME}/" \
    "${LOCAL_CHECKPOINT_DIR}/${NAME}/"
fi

echo "==> Done."
