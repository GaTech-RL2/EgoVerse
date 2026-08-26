#!/usr/bin/env bash
set -euo pipefail

# Pull Teleop Round (0724 robot manip data) checkpoints + dataset from Skynet.
# These are the NEAR-TABLE manipulation-phase policies trained on 28 robot teleop
# demos (penguin -> blue basket). Perception matches the robot exactly, so dropout
# was only 0.5 and all three are vision-driven (reliance ~x1.0). See
# ai_docs teleop rollout guide. Requires EgoVerse on branch rby1_aria_policy LATEST.
#
# Checkpoints (guide §1):
#   A  tel_resnet@999   logs/rby1_teleop_pp_0724/tel_resnet_1k/checkpoints/epoch_epoch=999.ckpt   clean 0.0191  x1.01  (primary; plain ResNet)
#   B  tel_wam3@899     logs/rby1_teleop_pp_0724_wam3/tel_wam3_1k/checkpoints/epoch_epoch=899.ckpt clean 0.0208  x1.02  (world-model head; best val)
#   C  tel_d3conv@999   logs/rby1_teleop_pp_0724/tel_d3conv_1k/checkpoints/epoch_epoch=999.ckpt   clean 0.0228  x1.06  (frozen DINOv3-S + ConvNeck)
#
# Usage:
#   bash pull_teleop_pp_0724.sh                    # A B C + dataset (full, ~195MB)
#   VARIANTS="A" bash pull_teleop_pp_0724.sh       # just the primary
#   SKIP_DATASET=1 bash pull_teleop_pp_0724.sh     # ckpts only
#   DATASET_SIZE=1 bash pull_teleop_pp_0724.sh     # just print remote dataset size + exit
#   EPOCH_A=699 bash pull_teleop_pp_0724.sh        # override an epoch
#   SKYNET_PASS=... bash pull_teleop_pp_0724.sh    # non-interactive (needs sshpass)

REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky2.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
VARIANTS="${VARIANTS:-A B C}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-22}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"

# Training corpus (needed for sim/HW GT-input eval + dataset_avg reset pose). Small (195MB).
REMOTE_DATASET="${REMOTE_DATASET:-${REMOTE_REPO}/datasets/rby1_teleop_pp_0724}"
LOCAL_DATASET="${LOCAL_DATASET:-./datasets/rby1_teleop_pp_0724}"

EPOCH_A="${EPOCH_A:-999}"
EPOCH_B="${EPOCH_B:-899}"
EPOCH_C="${EPOCH_C:-999}"

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
    A|a|tel_resnet|tel_resnet_1k)
      echo "A|tel_resnet@${EPOCH_A}|logs/rby1_teleop_pp_0724/tel_resnet_1k/checkpoints/epoch_epoch=${EPOCH_A}.ckpt|checkpoints/rby1_teleop_pp_0724/tel_resnet_1k/checkpoints/epoch_epoch=${EPOCH_A}.ckpt"
      ;;
    B|b|tel_wam3|tel_wam3_1k)
      echo "B|tel_wam3@${EPOCH_B}|logs/rby1_teleop_pp_0724_wam3/tel_wam3_1k/checkpoints/epoch_epoch=${EPOCH_B}.ckpt|checkpoints/rby1_teleop_pp_0724_wam3/tel_wam3_1k/checkpoints/epoch_epoch=${EPOCH_B}.ckpt"
      ;;
    C|c|tel_d3conv|tel_d3conv_1k)
      echo "C|tel_d3conv@${EPOCH_C}|logs/rby1_teleop_pp_0724/tel_d3conv_1k/checkpoints/epoch_epoch=${EPOCH_C}.ckpt|checkpoints/rby1_teleop_pp_0724/tel_d3conv_1k/checkpoints/epoch_epoch=${EPOCH_C}.ckpt"
      ;;
    *)
      return 1
      ;;
  esac
}

# Optional: just report remote dataset size and exit.
if [[ "${DATASET_SIZE:-0}" == "1" ]]; then
  echo "=== remote size: ${REMOTE_DATASET} ==="
  run_ssh "du -sh '${REMOTE_DATASET}' 2>/dev/null; echo '--- per subdir ---'; for d in meta data videos; do du -sh '${REMOTE_DATASET}'/\$d 2>/dev/null; done; echo '--- episodes ---'; ls -1 '${REMOTE_DATASET}'/data/*/ 2>/dev/null | wc -l"
  exit 0
fi

for v in $VARIANTS; do
  if ! entry="$(resolve_variant "$v")"; then
    echo "ERROR: unknown VARIANTS entry '$v'" >&2
    echo "  use: A B C  (or tel_resnet tel_wam3 tel_d3conv)" >&2
    exit 1
  fi
  IFS='|' read -r _tag label remote_rel local_rel <<<"$entry"
  pull_ckpt "$label" "$remote_rel" "$local_rel"
done

# --- dataset (full by default; 195MB LeRobot) ---------------------------------
if [[ "${SKIP_DATASET:-0}" != "1" ]]; then
  echo ""
  echo "=== dataset FULL: ${REMOTE_DATASET}/ -> ${LOCAL_DATASET}/ ==="
  run_rsync "${REMOTE_DATASET}/" "${LOCAL_DATASET}/"
fi

echo ""
echo "Done. Local checkpoints:"
for v in $VARIANTS; do
  entry="$(resolve_variant "$v")"
  IFS='|' read -r tag label _remote local_rel <<<"$entry"
  echo "  [${tag}] ${label}: ${local_rel}"
done
echo ""
echo "Next — sanity replay (serve A first, then):"
echo "  VARIANT=tel_resnet PORT=8000 bash serve_aria_egoposer.sh"
echo "  python egomimic/scripts/test_serve_policy_client.py --episode-idx 0 --max-steps 30 --trajectory \\"
echo "    --dataset-folder ${LOCAL_DATASET}"
echo ""
echo "Then sim GT (SEW side): GT_MODE=gt_proprio VARIANT=tel_resnet PORT=8000 \\"
echo "  DATASET=${PWD}/${LOCAL_DATASET#./} bash projects/rby1_teleop/run_rollout_aria_egoposer_sim.sh"
