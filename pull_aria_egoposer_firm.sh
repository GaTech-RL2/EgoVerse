#!/usr/bin/env bash
set -euo pipefail

# Pull the clean _firm aria_egoposer checkpoints (+ optional LeRobot dataset) from Skynet.
# Skynet uses password auth; you will be prompted once per transfer (or set SKYNET_PASS to
# use sshpass non-interactively).
#
# Checkpoints (VARIANTS):
#   v1            logs/aria_egoposer_firm/vanilla/checkpoints/last.ckpt
#   v2            logs/aria_egoposer_firm_v2/v2_hist_traj/checkpoints/last.ckpt
#   crop100_2k    logs/aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt  (R4 primary)
#   dino100_2k    logs/aria_egoposer_firm/dino100_2k/checkpoints/last.ckpt
#   dino_neck_2k  logs/aria_egoposer_firm/dino_neck_2k/checkpoints/last.ckpt  (experimental)
#   dino_lora_2k  logs/aria_egoposer_firm/dino_lora_2k/checkpoints/last.ckpt  (experimental)
#   r4            crop100_2k + dino100_2k
#
# Usage:
#   bash pull_aria_egoposer_firm.sh                  # v1 + v2 ckpts + dataset meta only
#   VARIANTS=r4 bash pull_aria_egoposer_firm.sh      # R4 primary + dino A/B ckpts
#   VARIANTS=crop100_2k bash pull_aria_egoposer_firm.sh
#   PULL_DATASET=1 bash pull_aria_egoposer_firm.sh   # also pull full LeRobot data/
#   PULL_CODE=1 VARIANTS=r4 bash pull_aria_egoposer_firm.sh  # + hpt.py / hpt_nets.py

# ====== config ======
REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky2.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
VARIANTS="${VARIANTS:-v1 v2}"   # v1 | v2 | both
REMOTE_DATASET="${REMOTE_DATASET:-${REMOTE_REPO}/datasets/aria_egoposer_firm}"
LOCAL_DATASET="${LOCAL_DATASET:-./datasets/aria_egoposer_firm}"
# SSH: script already forces password auth. If you see "Connection refused", Skynet
# port 22 is unreachable (GT VPN / on-campus / cluster down) — not missing -o flags.
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-22}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"
# ====================

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
  echo "=== ${label}: last.ckpt ==="
  run_rsync "${REMOTE_REPO}/${remote_rel}" "${local_rel}"
}

for v in $VARIANTS; do
  case "$v" in
    v1|firm|firm_v1|vanilla)
      pull_ckpt "V1 firm vanilla" \
        "logs/aria_egoposer_firm/vanilla/checkpoints/last.ckpt" \
        "checkpoints/aria_egoposer_firm/vanilla/checkpoints/last.ckpt"
      ;;
    v2|firm_v2|hist_traj|v2_hist_traj)
      pull_ckpt "V2 firm hist+traj" \
        "logs/aria_egoposer_firm_v2/v2_hist_traj/checkpoints/last.ckpt" \
        "checkpoints/aria_egoposer_firm_v2/v2_hist_traj/checkpoints/last.ckpt"
      ;;
    crop100_2k|crop100|r4_primary)
      pull_ckpt "R4 crop100_2k (primary)" \
        "logs/aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt" \
        "checkpoints/aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt"
      ;;
    dino100_2k|dino100)
      pull_ckpt "R4 dino100_2k" \
        "logs/aria_egoposer_firm/dino100_2k/checkpoints/last.ckpt" \
        "checkpoints/aria_egoposer_firm/dino100_2k/checkpoints/last.ckpt"
      ;;
    dino_neck_2k|dino_neck)
      pull_ckpt "R4 dino_neck_2k (experimental)" \
        "logs/aria_egoposer_firm/dino_neck_2k/checkpoints/last.ckpt" \
        "checkpoints/aria_egoposer_firm/dino_neck_2k/checkpoints/last.ckpt"
      ;;
    dino_lora_2k|dino_lora)
      pull_ckpt "R4 dino_lora_2k (experimental)" \
        "logs/aria_egoposer_firm/dino_lora_2k/checkpoints/last.ckpt" \
        "checkpoints/aria_egoposer_firm/dino_lora_2k/checkpoints/last.ckpt"
      ;;
    r4|r4_deploy)
      pull_ckpt "R4 crop100_2k (primary)" \
        "logs/aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt" \
        "checkpoints/aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt"
      pull_ckpt "R4 dino100_2k" \
        "logs/aria_egoposer_firm/dino100_2k/checkpoints/last.ckpt" \
        "checkpoints/aria_egoposer_firm/dino100_2k/checkpoints/last.ckpt"
      ;;
    *)
      echo "ERROR: unknown VARIANTS entry '$v'" >&2
      echo "  use: v1 v2 crop100_2k dino100_2k dino_neck_2k dino_lora_2k r4" >&2
      exit 1
      ;;
  esac
done

if [[ "${PULL_CODE:-0}" == "1" ]]; then
  echo "=== R4 required source (proprio_clamp + DINOv2/LoRALinear) ==="
  run_rsync "${REMOTE_REPO}/egomimic/algo/hpt.py" "egomimic/algo/hpt.py"
  run_rsync "${REMOTE_REPO}/egomimic/models/hpt_nets.py" "egomimic/models/hpt_nets.py"
fi

if [[ "${PULL_HF_CACHE:-0}" == "1" ]]; then
  echo "=== HuggingFace cache (DINO weights; large) ==="
  REMOTE_HF="${REMOTE_HF:-/coc/flash7/czhang883/.cache/huggingface}"
  LOCAL_HF="${LOCAL_HF:-${HOME}/.cache/huggingface}"
  mkdir -p "${LOCAL_HF}"
  run_rsync "${REMOTE_HF}/" "${LOCAL_HF}/"
fi

if [[ "${PULL_DATASET:-0}" == "1" ]]; then
  echo "=== full LeRobot dataset (data/ + meta/, needed for reset + GT modes) ==="
  echo "    remote: ${REMOTE_DATASET}/"
  echo "    local:  ${LOCAL_DATASET}/"
  run_rsync "${REMOTE_DATASET}/" "${LOCAL_DATASET}/"
else
  echo "=== dataset meta only (info/stats/episodes/tasks) ==="
  echo "    (set PULL_DATASET=1 to also pull data/ for dataset_avg reset + GT replay)"
  run_rsync "${REMOTE_DATASET}/meta/" "${LOCAL_DATASET}/meta/"
fi

echo ""
echo "Done. Checkpoints:"
for v in $VARIANTS; do
  case "$v" in
    v1|firm|firm_v1|vanilla)
      echo "  V1: checkpoints/aria_egoposer_firm/vanilla/checkpoints/last.ckpt"
      ;;
    v2|firm_v2|hist_traj|v2_hist_traj)
      echo "  V2: checkpoints/aria_egoposer_firm_v2/v2_hist_traj/checkpoints/last.ckpt"
      ;;
    crop100_2k|crop100|r4_primary|r4|r4_deploy)
      echo "  R4 crop100_2k: checkpoints/aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt"
      ;;
    dino100_2k|dino100|r4|r4_deploy)
      echo "  R4 dino100_2k: checkpoints/aria_egoposer_firm/dino100_2k/checkpoints/last.ckpt"
      ;;
    dino_neck_2k|dino_neck)
      echo "  R4 dino_neck_2k: checkpoints/aria_egoposer_firm/dino_neck_2k/checkpoints/last.ckpt"
      ;;
    dino_lora_2k|dino_lora)
      echo "  R4 dino_lora_2k: checkpoints/aria_egoposer_firm/dino_lora_2k/checkpoints/last.ckpt"
      ;;
  esac
done
echo "Dataset: ${LOCAL_DATASET}/"
echo ""
echo "Next:"
echo "  source emimic/bin/activate"
echo "  python inspect_aria_egoposer.py checkpoints/aria_egoposer_firm/vanilla/checkpoints/last.ckpt"
echo "  VARIANT=firm_v1 bash serve_aria_egoposer.sh"
