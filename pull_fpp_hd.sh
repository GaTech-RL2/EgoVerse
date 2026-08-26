#!/usr/bin/env bash
set -euo pipefail

# Pull FPP HD-era round (2026-07-21 evening) checkpoints from Skynet.
# These are proprio-dropout-0.9 (vision-driven) runs; pull the EXACT gate-verified
# epochs from the guide's ranked table (higher epoch is NOT necessarily better here).
# Requires EgoVerse on branch rby1_aria_policy LATEST (needs egomimic/utils/image_augs.py
# and egomimic/models/custom_encoders.py with the nvs3d cold-load fix to unpickle).
#
# Ranked list (guide §1):
#   A  hd_wam3@1399    logs/aria_fullpp_wam3/fpp_hd_wam3_2k/checkpoints/epoch_epoch=1399.ckpt   clean 0.025 reliance x1.03
#   B  hd_resnet@1499  logs/aria_fullpp/fpp_hd_resnet_2k/checkpoints/epoch_epoch=1499.ckpt      clean 0.024 reliance x1.00
#   C  wam3@1599       logs/aria_fullpp_wam3/fpp_wam3_2k/checkpoints/epoch_epoch=1599.ckpt       0.6-era baseline (proprio x1.28)
#
# NOTE: guide §2 shows A at epoch 899 (inconsistent with the §1 table's 1399). Default
# here is 1399; override with EPOCH_A=899 if the author confirms.
#
# Usage:
#   bash pull_fpp_hd.sh                          # A B C (all three)
#   VARIANTS="A B" bash pull_fpp_hd.sh           # A/B only
#   EPOCH_A=899 bash pull_fpp_hd.sh              # override A epoch
#   EPOCH_C=1499 bash pull_fpp_hd.sh             # if 1599 not on remote yet, fall back
#   SKYNET_PASS=... bash pull_fpp_hd.sh          # non-interactive (needs sshpass)

REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky2.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
VARIANTS="${VARIANTS:-A B C}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-22}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"

# HD training corpus (needed for GT-input eval S1/S2 — the policy input in gt_proprio).
# aria_egoposer_firm on this machine is the OLD firm round, NOT what HD trained on.
REMOTE_DATASET="${REMOTE_DATASET:-${REMOTE_REPO}/datasets/aria_fullpp}"
LOCAL_DATASET="${LOCAL_DATASET:-./datasets/aria_fullpp}"

EPOCH_A="${EPOCH_A:-1399}"
EPOCH_B="${EPOCH_B:-1499}"
EPOCH_C="${EPOCH_C:-1599}"

RSYNC_BIN="/usr/bin/rsync"
[[ -x "$RSYNC_BIN" ]] || RSYNC_BIN="$(command -v rsync)"

_ssh_rsync_cmd() {
  # shellcheck disable=SC2086
  printf '%s' "ssh -p ${REMOTE_SSH_PORT} -o PreferredAuthentications=password -o PubkeyAuthentication=no -o IdentitiesOnly=yes ${SSH_EXTRA_OPTS}"
}

run_ssh() {
  local remote_cmd="$1"
  # shellcheck disable=SC2086
  if [[ -n "${SKYNET_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      sshpass -p "$SKYNET_PASS" $(_ssh_rsync_cmd) "${REMOTE_USER_HOST}" "${remote_cmd}"
  else
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      $(_ssh_rsync_cmd) "${REMOTE_USER_HOST}" "${remote_cmd}"
  fi
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

pull_ckpt() {
  local label="$1" remote_rel="$2" local_rel="$3"
  echo "=== ${label} ==="
  echo "    remote: ${REMOTE_REPO}/${remote_rel}"
  echo "    local:  ${local_rel}"
  run_rsync "${REMOTE_REPO}/${remote_rel}" "${local_rel}"
}

# tag|label|remote_rel|local_rel
resolve_variant() {
  local v="$1"
  case "$v" in
    A|a|hd_wam3|hd_wam3_2k)
      echo "A|hd_wam3@${EPOCH_A}|logs/aria_fullpp_wam3/fpp_hd_wam3_2k/checkpoints/epoch_epoch=${EPOCH_A}.ckpt|checkpoints/aria_fullpp_wam3/fpp_hd_wam3_2k/checkpoints/epoch_epoch=${EPOCH_A}.ckpt"
      ;;
    B|b|hd_resnet|hd_resnet_2k)
      echo "B|hd_resnet@${EPOCH_B}|logs/aria_fullpp/fpp_hd_resnet_2k/checkpoints/epoch_epoch=${EPOCH_B}.ckpt|checkpoints/aria_fullpp/fpp_hd_resnet_2k/checkpoints/epoch_epoch=${EPOCH_B}.ckpt"
      ;;
    C|c|wam3|wam3_2k)
      echo "C|wam3@${EPOCH_C}|logs/aria_fullpp_wam3/fpp_wam3_2k/checkpoints/epoch_epoch=${EPOCH_C}.ckpt|checkpoints/aria_fullpp_wam3/fpp_wam3_2k/checkpoints/epoch_epoch=${EPOCH_C}.ckpt"
      ;;
    *)
      return 1
      ;;
  esac
}

for v in $VARIANTS; do
  if ! entry="$(resolve_variant "$v")"; then
    echo "ERROR: unknown VARIANTS entry '$v'" >&2
    echo "  use: A B C  (or hd_wam3 hd_resnet wam3)" >&2
    exit 1
  fi
  IFS='|' read -r _tag label remote_rel local_rel <<<"$entry"
  pull_ckpt "$label" "$remote_rel" "$local_rel"
done

# --- aria_fullpp dataset (HD training corpus) ---------------------------------
# DATASET_SIZE=1     : print remote du -sh of aria_fullpp (and meta/ + data/ + videos/) and exit dataset step
# PULL_DATASET=1     : pull meta/ only (default dataset action) — small, lets you inspect episode list
# PULL_DATASET_FULL=1: pull the entire dataset (meta + data + videos) — can be many GB
if [[ "${DATASET_SIZE:-0}" == "1" ]]; then
  echo ""
  echo "=== remote size: ${REMOTE_DATASET} ==="
  run_ssh "du -sh '${REMOTE_DATASET}' 2>/dev/null; echo '--- per subdir ---'; for d in meta data videos; do du -sh '${REMOTE_DATASET}'/\$d 2>/dev/null; done; echo '--- episodes ---'; ls -1 '${REMOTE_DATASET}'/data/*/ 2>/dev/null | wc -l"
fi

if [[ "${PULL_DATASET_FULL:-0}" == "1" ]]; then
  echo ""
  echo "=== aria_fullpp FULL: ${REMOTE_DATASET}/ -> ${LOCAL_DATASET}/ ==="
  run_rsync "${REMOTE_DATASET}/" "${LOCAL_DATASET}/"
elif [[ "${PULL_DATASET:-0}" == "1" ]]; then
  echo ""
  echo "=== aria_fullpp META only (set PULL_DATASET_FULL=1 for data+videos): ${REMOTE_DATASET}/meta/ ==="
  run_rsync "${REMOTE_DATASET}/meta/" "${LOCAL_DATASET}/meta/"
fi

echo ""
echo "Done. Local checkpoints:"
for v in $VARIANTS; do
  entry="$(resolve_variant "$v")"
  IFS='|' read -r tag label _remote local_rel <<<"$entry"
  echo "  [${tag}] ${label}: ${local_rel}"
done
echo ""
echo "Next (serve A first, then rollout with head frozen):"
echo "  git pull && source emimic/bin/activate"
echo "  VARIANT=hd_wam3 PORT=8000 bash serve_aria_egoposer.sh"
echo "  # then on the SEW-Geometric-Teleop side:"
echo "  # FREEZE_HEAD=1 SAFE_MODE=1 SHOW_CAMERA=1 EXEC_STEPS=16 FREQ=10 PORT=8000 \\"
echo "  #   DATASET=/home/aloha/RB_Y1_workspace/EgoVerse/datasets/aria_egoposer_firm \\"
echo "  #   bash projects/rby1_teleop/run_rollout_aria_egoposer.sh"
