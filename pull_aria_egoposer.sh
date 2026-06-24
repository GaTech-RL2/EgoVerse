#!/usr/bin/env bash
set -euo pipefail

# Pull the aria_egoposer whole-body image policy checkpoints (+ dataset meta) from Skynet.
# Skynet uses password auth; you will be prompted once per transfer (or set SKYNET_PASS to
# use sshpass non-interactively).
#
# Usage:
#   bash pull_aria_egoposer.sh            # hier last.ckpt + vanilla last.ckpt + meta json
#   VARIANTS="hier" bash pull_aria_egoposer.sh
#   EPOCHS="599 999" bash pull_aria_egoposer.sh   # also grab specific earlier epochs

# ====== config ======
REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky1.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
LOCAL_ROOT="${LOCAL_ROOT:-./checkpoints/aria_egoposer}"
VARIANTS="${VARIANTS:-hier vanilla}"
EPOCHS="${EPOCHS:-}"   # space-separated epoch numbers, e.g. "599 999"; empty = last.ckpt only
# ====================

RSYNC_BIN="/usr/bin/rsync"
[[ -x "$RSYNC_BIN" ]] || RSYNC_BIN="$(command -v rsync)"

run_rsync() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -n "${SKYNET_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      sshpass -p "$SKYNET_PASS" \
      "$RSYNC_BIN" -avh --progress --partial --inplace \
      -e "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no" \
      "${REMOTE_USER_HOST}:${src}" "$dst"
  else
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      "$RSYNC_BIN" -avh --progress --partial --inplace \
      -e "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no" \
      "${REMOTE_USER_HOST}:${src}" "$dst"
  fi
}

for v in $VARIANTS; do
  echo "=== ${v}: last.ckpt ==="
  run_rsync "${REMOTE_REPO}/logs/aria_egoposer/${v}/checkpoints/last.ckpt" \
            "${LOCAL_ROOT}/${v}/checkpoints/last.ckpt"
  for e in $EPOCHS; do
    echo "=== ${v}: epoch_epoch=${e}.ckpt ==="
    run_rsync "${REMOTE_REPO}/logs/aria_egoposer/${v}/checkpoints/epoch_epoch=${e}.ckpt" \
              "${LOCAL_ROOT}/${v}/checkpoints/epoch_epoch=${e}.ckpt"
  done
done

if [[ "${PULL_DATASET:-0}" == "1" ]]; then
  echo "=== full LeRobot dataset (data/ + meta/, needed for dataset_avg reset) ==="
  run_rsync "${REMOTE_REPO}/datasets/aria_egoposer/" \
            "./datasets/aria_egoposer/"
else
  echo "=== dataset meta only (info/stats/episodes/tasks) ==="
  echo "    (set PULL_DATASET=1 to also pull data/ for the dataset_avg reset)"
  run_rsync "${REMOTE_REPO}/datasets/aria_egoposer/meta/" \
            "./datasets/aria_egoposer/meta/"
fi

echo ""
echo "Done. Recommended checkpoint:"
echo "  ${LOCAL_ROOT}/hier/checkpoints/last.ckpt"
