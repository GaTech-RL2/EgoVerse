#!/usr/bin/env bash
set -euo pipefail

# Pull FPP Round (2026-07-20) checkpoints from Skynet.
# Requires EgoVerse checkout on branch rby1_aria_policy @ dd74911d or later
# (new image_augs classes must be present to unpickle).
#
# Priority tags (from deployment note):
#   A  d3lora@99     logs/aria_fullpp/fpp_d3lora_2k/checkpoints/epoch_epoch=99.ckpt
#   B  d3conv@399    logs/aria_fullpp/fpp_d3conv_2k/checkpoints/epoch_epoch=399.ckpt
#   C  wam3@999      logs/aria_fullpp_wam3/fpp_wam3_2k/checkpoints/epoch_epoch=999.ckpt
#   D  resnet@1599   logs/aria_fullpp/fpp_resnet_2k/checkpoints/epoch_epoch=1599.ckpt
#   E  bare@1399     logs/exp1_bare/fpp_bare_2k/checkpoints/epoch_epoch=1399.ckpt   (opt)
#   F  glove@699     logs/exp1_glove/fpp_glove_2k/checkpoints/epoch_epoch=699.ckpt  (opt)
#
# Usage:
#   bash pull_fpp_round.sh                         # A–D (priority set)
#   VARIANTS="A B C D E F" bash pull_fpp_round.sh  # all including optional
#   VARIANTS=d3lora bash pull_fpp_round.sh         # single by tag/name
#   PULL_CODE=1 bash pull_fpp_round.sh             # also sync image_augs / related code
#   PULL_VAL=1 bash pull_fpp_round.sh              # also pull rby1_teleop_val_v2 meta(+data)
#
# Same-stage A/B (e.g. all @ ~1299): see pull_fpp_same_epoch.sh
#   LIST_ONLY=1 bash pull_fpp_same_epoch.sh
#   TARGET_EPOCH=1299 VARIANTS="A B C D E F" bash pull_fpp_same_epoch.sh


REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky2.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
VARIANTS="${VARIANTS:-A B C D}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-22}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"
REMOTE_VAL="${REMOTE_VAL:-${REMOTE_REPO}/datasets/rby1_teleop_val_v2}"
LOCAL_VAL="${LOCAL_VAL:-./datasets/rby1_teleop_val_v2}"

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

pull_ckpt() {
  local label="$1" remote_rel="$2" local_rel="$3"
  echo "=== ${label} ==="
  echo "    remote: ${REMOTE_REPO}/${remote_rel}"
  echo "    local:  ${local_rel}"
  run_rsync "${REMOTE_REPO}/${remote_rel}" "${local_rel}"
}

resolve_variant() {
  local v="$1"
  case "$v" in
    A|a|d3lora|d3lora_2k|d3lora@99)
      echo "A|d3lora@99|logs/aria_fullpp/fpp_d3lora_2k/checkpoints/epoch_epoch=99.ckpt|checkpoints/aria_fullpp/fpp_d3lora_2k/checkpoints/epoch_epoch=99.ckpt"
      ;;
    A399|d3lora@399)
      echo "A399|d3lora@399|logs/aria_fullpp/fpp_d3lora_2k/checkpoints/epoch_epoch=399.ckpt|checkpoints/aria_fullpp/fpp_d3lora_2k/checkpoints/epoch_epoch=399.ckpt"
      ;;
    B|b|d3conv|d3conv_2k|d3conv@399)
      echo "B|d3conv@399|logs/aria_fullpp/fpp_d3conv_2k/checkpoints/epoch_epoch=399.ckpt|checkpoints/aria_fullpp/fpp_d3conv_2k/checkpoints/epoch_epoch=399.ckpt"
      ;;
    C|c|wam3|wam3_2k|wam3@999)
      echo "C|wam3@999|logs/aria_fullpp_wam3/fpp_wam3_2k/checkpoints/epoch_epoch=999.ckpt|checkpoints/aria_fullpp_wam3/fpp_wam3_2k/checkpoints/epoch_epoch=999.ckpt"
      ;;
    D|d|resnet|resnet_2k|resnet@1599)
      echo "D|resnet@1599|logs/aria_fullpp/fpp_resnet_2k/checkpoints/epoch_epoch=1599.ckpt|checkpoints/aria_fullpp/fpp_resnet_2k/checkpoints/epoch_epoch=1599.ckpt"
      ;;
    E|e|bare|bare_2k|bare@1399)
      echo "E|bare@1399|logs/exp1_bare/fpp_bare_2k/checkpoints/epoch_epoch=1399.ckpt|checkpoints/exp1_bare/fpp_bare_2k/checkpoints/epoch_epoch=1399.ckpt"
      ;;
    F|f|glove|glove_2k|glove@699)
      echo "F|glove@699|logs/exp1_glove/fpp_glove_2k/checkpoints/epoch_epoch=699.ckpt|checkpoints/exp1_glove/fpp_glove_2k/checkpoints/epoch_epoch=699.ckpt"
      ;;
    *)
      return 1
      ;;
  esac
}

for v in $VARIANTS; do
  if ! entry="$(resolve_variant "$v")"; then
    echo "ERROR: unknown VARIANTS entry '$v'" >&2
    echo "  use: A B C D E F  (or d3lora d3conv wam3 resnet bare glove)" >&2
    exit 1
  fi
  IFS='|' read -r _tag label remote_rel local_rel <<<"$entry"
  pull_ckpt "$label" "$remote_rel" "$local_rel"
done

if [[ "${PULL_CODE:-0}" == "1" ]]; then
  echo "=== FPP-required source (image_augs + related; prefer git checkout of rby1_aria_policy) ==="
  run_rsync "${REMOTE_REPO}/egomimic/utils/image_augs.py" "egomimic/utils/image_augs.py"
  # Best-effort: these often changed with FPP; ignore failures if absent remotely.
  run_rsync "${REMOTE_REPO}/egomimic/algo/hpt.py" "egomimic/algo/hpt.py" || true
  run_rsync "${REMOTE_REPO}/egomimic/models/hpt_nets.py" "egomimic/models/hpt_nets.py" || true
fi

if [[ "${PULL_VAL:-0}" == "1" ]]; then
  if [[ "${PULL_VAL_FULL:-0}" == "1" ]]; then
    echo "=== val dataset full: ${REMOTE_VAL}/ -> ${LOCAL_VAL}/ ==="
    run_rsync "${REMOTE_VAL}/" "${LOCAL_VAL}/"
  else
    echo "=== val dataset meta only (set PULL_VAL_FULL=1 for data/) ==="
    run_rsync "${REMOTE_VAL}/meta/" "${LOCAL_VAL}/meta/"
  fi
fi

echo ""
echo "Done. Local checkpoints:"
for v in $VARIANTS; do
  entry="$(resolve_variant "$v")"
  IFS='|' read -r tag label _remote local_rel <<<"$entry"
  echo "  [${tag}] ${label}: ${local_rel}"
done
echo ""
echo "Next (serve A first):"
echo "  # ensure branch: git fetch && git checkout rby1_aria_policy && git pull"
echo "  source emimic/bin/activate"
echo "  VARIANT=d3lora bash serve_aria_egoposer.sh"
echo "  # or: CKPT=checkpoints/aria_fullpp/fpp_d3lora_2k/checkpoints/epoch_epoch=99.ckpt bash serve_aria_egoposer.sh"
