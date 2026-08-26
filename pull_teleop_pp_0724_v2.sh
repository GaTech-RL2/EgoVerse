#!/usr/bin/env bash
set -euo pipefail

# Pull Teleop Round v2 (Savitzky–Golay action targets, proprio dropout 0.9).
# Primary: epoch_epoch=1499.ckpt  | alts: 999, 1999
#
# Usage:
#   bash pull_teleop_pp_0724_v2.sh                 # primary@1499 + SG dataset
#   EPOCHS="999 1499 1999" bash pull_teleop_pp_0724_v2.sh
#   SKIP_DATASET=1 bash pull_teleop_pp_0724_v2.sh
#   SKYNET_PASS=... bash pull_teleop_pp_0724_v2.sh

REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky2.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-22}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"

RUN_REL="logs/RBY1_wb_img_tel_v2/wb_img_pickplace_v2_2k/checkpoints"
LOCAL_CKPT_DIR="./checkpoints/RBY1_wb_img_tel_v2/wb_img_pickplace_v2_2k/checkpoints"
EPOCHS="${EPOCHS:-1499}"

REMOTE_DATASET="${REMOTE_DATASET:-${REMOTE_REPO}/datasets/rby1_teleop_pp_0724_sg}"
LOCAL_DATASET="${LOCAL_DATASET:-./datasets/rby1_teleop_pp_0724_sg}"

RSYNC_BIN="/usr/bin/rsync"
[[ -x "$RSYNC_BIN" ]] || RSYNC_BIN="$(command -v rsync)"

_ssh_rsync_cmd() {
  # shellcheck disable=SC2086
  printf '%s' "ssh -p ${REMOTE_SSH_PORT} -o PreferredAuthentications=password -o PubkeyAuthentication=no -o IdentitiesOnly=yes ${SSH_EXTRA_OPTS}"
}

run_rsync() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  local ssh_cmd
  ssh_cmd="$(_ssh_rsync_cmd)"
  if [[ -n "${SKYNET_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      sshpass -p "$SKYNET_PASS" \
      "$RSYNC_BIN" -avh --progress --partial --inplace \
      -e "$ssh_cmd" \
      "${REMOTE_USER_HOST}:${src}" "$dst"
  else
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      "$RSYNC_BIN" -avh --progress --partial --inplace \
      -e "$ssh_cmd" \
      "${REMOTE_USER_HOST}:${src}" "$dst"
  fi
}

for ep in $EPOCHS; do
  name="epoch_epoch=${ep}.ckpt"
  echo "=== v2 @${ep} ==="
  run_rsync "${REMOTE_REPO}/${RUN_REL}/${name}" "${LOCAL_CKPT_DIR}/${name}"
done

if [[ "${SKIP_DATASET:-0}" != "1" ]]; then
  echo ""
  echo "=== dataset: ${REMOTE_DATASET}/ -> ${LOCAL_DATASET}/ ==="
  run_rsync "${REMOTE_DATASET}/" "${LOCAL_DATASET}/"
fi

echo ""
echo "Done. Local:"
ls -lh "${LOCAL_CKPT_DIR}"/epoch_epoch=*.ckpt 2>/dev/null || true
echo "Dataset: ${LOCAL_DATASET}"
