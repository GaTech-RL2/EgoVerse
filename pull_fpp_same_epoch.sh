#!/usr/bin/env bash
set -euo pipefail

# Pull the SAME training stage across FPP variants (default target epoch 1299).
# If epoch_epoch=${TARGET}.ckpt is missing, picks the nearest available epoch
# (ties prefer the higher epoch). Use LIST_ONLY=1 to just print the pick table.
#
# Runs (same as pull_fpp_round.sh):
#   A d3lora   logs/aria_fullpp/fpp_d3lora_2k
#   B d3conv   logs/aria_fullpp/fpp_d3conv_2k
#   C wam3     logs/aria_fullpp_wam3/fpp_wam3_2k
#   D resnet   logs/aria_fullpp/fpp_resnet_2k
#   E bare     logs/exp1_bare/fpp_bare_2k
#   F glove    logs/exp1_glove/fpp_glove_2k
#
# Usage:
#   bash pull_fpp_same_epoch.sh                         # A–D @ ~1299
#   VARIANTS="A B C D E F" bash pull_fpp_same_epoch.sh  # all six
#   TARGET_EPOCH=999 bash pull_fpp_same_epoch.sh        # shared stage 999
#   STRICT=1 TARGET_EPOCH=1299 bash pull_fpp_same_epoch.sh  # fail if exact missing
#   LIST_ONLY=1 bash pull_fpp_same_epoch.sh             # dry-run: list + nearest

REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky2.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
VARIANTS="${VARIANTS:-A B C D}"
TARGET_EPOCH="${TARGET_EPOCH:-1299}"
STRICT="${STRICT:-0}"
LIST_ONLY="${LIST_ONLY:-0}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-22}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"

RSYNC_BIN="/usr/bin/rsync"
[[ -x "$RSYNC_BIN" ]] || RSYNC_BIN="$(command -v rsync)"

_ssh_base() {
  # shellcheck disable=SC2086
  echo ssh -p "${REMOTE_SSH_PORT}" \
    -o PreferredAuthentications=password \
    -o PubkeyAuthentication=no \
    -o IdentitiesOnly=yes \
    ${SSH_EXTRA_OPTS}
}

run_ssh() {
  local remote_cmd="$1"
  # shellcheck disable=SC2046
  if [[ -n "${SKYNET_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      sshpass -p "$SKYNET_PASS" \
      $(_ssh_base) "${REMOTE_USER_HOST}" "${remote_cmd}"
  else
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      $(_ssh_base) "${REMOTE_USER_HOST}" "${remote_cmd}"
  fi
}

run_rsync() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  local ssh_cmd
  ssh_cmd="$(_ssh_base | tr '\n' ' ')"
  if [[ -n "${SKYNET_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      sshpass -p "$SKYNET_PASS" \
      "$RSYNC_BIN" -avh --progress --partial --inplace \
      -e "${ssh_cmd}" \
      "${REMOTE_USER_HOST}:${src}" "$dst"
  else
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      "$RSYNC_BIN" -avh --progress --partial --inplace \
      -e "${ssh_cmd}" \
      "${REMOTE_USER_HOST}:${src}" "$dst"
  fi
}

# tag|name|remote_run_rel|local_run_rel
resolve_run() {
  local v="$1"
  case "$v" in
    A|a|d3lora|d3lora_2k)
      echo "A|d3lora|logs/aria_fullpp/fpp_d3lora_2k|checkpoints/aria_fullpp/fpp_d3lora_2k"
      ;;
    B|b|d3conv|d3conv_2k)
      echo "B|d3conv|logs/aria_fullpp/fpp_d3conv_2k|checkpoints/aria_fullpp/fpp_d3conv_2k"
      ;;
    C|c|wam3|wam3_2k)
      echo "C|wam3|logs/aria_fullpp_wam3/fpp_wam3_2k|checkpoints/aria_fullpp_wam3/fpp_wam3_2k"
      ;;
    D|d|resnet|resnet_2k)
      echo "D|resnet|logs/aria_fullpp/fpp_resnet_2k|checkpoints/aria_fullpp/fpp_resnet_2k"
      ;;
    E|e|bare|bare_2k)
      echo "E|bare|logs/exp1_bare/fpp_bare_2k|checkpoints/exp1_bare/fpp_bare_2k"
      ;;
    F|f|glove|glove_2k)
      echo "F|glove|logs/exp1_glove/fpp_glove_2k|checkpoints/exp1_glove/fpp_glove_2k"
      ;;
    *)
      return 1
      ;;
  esac
}

list_remote_epochs() {
  local remote_run="$1"
  local ckpt_dir="${REMOTE_REPO}/${remote_run}/checkpoints"
  # print epoch numbers only, one per line, sorted
  run_ssh "ls -1 '${ckpt_dir}'/epoch_epoch=*.ckpt 2>/dev/null | sed -n 's/.*epoch_epoch=\\([0-9]\\+\\)\\.ckpt/\\1/p' | sort -n"
}

pick_nearest_epoch() {
  local target="$1"
  local epochs_str="$2"
  local best="" best_dist=""
  local e dist
  while IFS= read -r e; do
    [[ -z "$e" ]] && continue
    dist=$(( e > target ? e - target : target - e ))
    if [[ -z "$best" ]] || (( dist < best_dist )) || (( dist == best_dist && e > best )); then
      best="$e"
      best_dist="$dist"
    fi
  done <<<"$epochs_str"
  if [[ -z "$best" ]]; then
    return 1
  fi
  echo "$best"
}

echo "Target shared stage: epoch ${TARGET_EPOCH}  (STRICT=${STRICT}, LIST_ONLY=${LIST_ONLY})"
echo "Remote: ${REMOTE_USER_HOST}:${REMOTE_REPO}"
echo ""

declare -a SUMMARY=()

for v in $VARIANTS; do
  if ! entry="$(resolve_run "$v")"; then
    echo "ERROR: unknown VARIANTS entry '$v'" >&2
    echo "  use: A B C D E F  (or d3lora d3conv wam3 resnet bare glove)" >&2
    exit 1
  fi
  IFS='|' read -r tag name remote_run local_run <<<"$entry"

  echo "=== [${tag}] ${name} ==="
  echo "    listing ${remote_run}/checkpoints ..."
  epochs="$(list_remote_epochs "$remote_run" || true)"
  if [[ -z "${epochs//[$'\n']/}" ]]; then
    echo "    ERROR: no epoch_epoch=*.ckpt found remotely" >&2
    SUMMARY+=("[${tag}] ${name}: MISSING (no ckpts)")
    if [[ "$STRICT" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  n_ckpts="$(printf '%s\n' "$epochs" | grep -c . || true)"
  min_e="$(printf '%s\n' "$epochs" | head -1)"
  max_e="$(printf '%s\n' "$epochs" | tail -1)"
  echo "    found ${n_ckpts} ckpts (min=${min_e}, max=${max_e})"

  if printf '%s\n' "$epochs" | grep -qx "${TARGET_EPOCH}"; then
    chosen="${TARGET_EPOCH}"
    note="exact"
  else
    if [[ "$STRICT" == "1" ]]; then
      echo "    ERROR: epoch ${TARGET_EPOCH} missing and STRICT=1" >&2
      SUMMARY+=("[${tag}] ${name}: MISSING exact @${TARGET_EPOCH} (range ${min_e}-${max_e})")
      exit 1
    fi
    chosen="$(pick_nearest_epoch "$TARGET_EPOCH" "$epochs")"
    note="nearest→${chosen} (wanted ${TARGET_EPOCH})"
  fi

  remote_rel="${remote_run}/checkpoints/epoch_epoch=${chosen}.ckpt"
  local_rel="${local_run}/checkpoints/epoch_epoch=${chosen}.ckpt"
  echo "    pick: epoch ${chosen} (${note})"
  echo "    remote: ${REMOTE_REPO}/${remote_rel}"
  echo "    local:  ${local_rel}"
  SUMMARY+=("[${tag}] ${name}: epoch ${chosen} (${note})")

  if [[ "$LIST_ONLY" == "1" ]]; then
    continue
  fi
  run_rsync "${REMOTE_REPO}/${remote_rel}" "${local_rel}"
done

echo ""
echo "==== same-stage pull summary (target ${TARGET_EPOCH}) ===="
for line in "${SUMMARY[@]}"; do
  echo "  ${line}"
done

if [[ "$LIST_ONLY" == "1" ]]; then
  echo ""
  echo "Dry-run only. Re-run without LIST_ONLY=1 to download."
  exit 0
fi

echo ""
echo "Serve examples (use explicit CKPT path — epochs may differ per run):"
for v in $VARIANTS; do
  entry="$(resolve_run "$v")"
  IFS='|' read -r tag name _remote local_run <<<"$entry"
  # best-effort: show whatever we just documented in SUMMARY via glob
  ckpt="$(ls -1 "${local_run}/checkpoints"/epoch_epoch=*.ckpt 2>/dev/null | sort -t= -k2 -n | tail -1 || true)"
  if [[ -n "$ckpt" ]]; then
    echo "  # [${tag}] ${name}"
    echo "  CKPT=${ckpt} PORT=8000 bash serve_aria_egoposer.sh"
  fi
done
