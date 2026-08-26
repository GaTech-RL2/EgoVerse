#!/usr/bin/env bash
set -euo pipefail

# Pull the point-cloud / depth policies (DP3 + Adapt3R) from Skynet:
#   1. checkpoints  (DP3 1024 @1299 primary, @1999 fallback; Adapt3R @1999)
#   2. LeRobot datasets (pcd1024_glass ~95MB, slamrect_rgbd ~850MB)
#   3. a *.py/*.yaml snapshot of the remote egomimic/ tree -> ./skynet_snapshot/
#      (the DP3/Adapt3R model + serving code exists ONLY in the Skynet working
#       tree — it is not on the GitHub rby1_aria_policy branch. After pulling,
#       run:  python apply_skynet_snapshot.py            # dry-run report
#             python apply_skynet_snapshot.py --apply    # copy new+changed files)
#
# Usage:
#   cd ~/RB_Y1_workspace/EgoVerse
#   read -rs -p "Skynet pw: " SKYNET_PASS; export SKYNET_PASS; echo
#   bash pull_dp3_adapt3r.sh
#
#   WITH_2048=1 bash pull_dp3_adapt3r.sh    # also DP3-2048 ckpt + dataset (~190MB)
#   SKIP_DATASETS=1 bash pull_dp3_adapt3r.sh
#   SKIP_CODE=1 bash pull_dp3_adapt3r.sh

REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky2.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-22}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"

WITH_2048="${WITH_2048:-0}"
SKIP_DATASETS="${SKIP_DATASETS:-0}"
SKIP_CODE="${SKIP_CODE:-0}"

RSYNC_BIN="/usr/bin/rsync"
[[ -x "$RSYNC_BIN" ]] || RSYNC_BIN="$(command -v rsync)"

_ssh_rsync_cmd() {
  # shellcheck disable=SC2086
  printf '%s' "ssh -p ${REMOTE_SSH_PORT} -i $HOME/.ssh/id_ed25519_czhang_skynet -o IdentitiesOnly=yes -o PreferredAuthentications=publickey -o ServerAliveInterval=30 -o ServerAliveCountMax=6 ${SSH_EXTRA_OPTS}"
}

run_rsync() {
  local src="$1" dst="$2"; shift 2
  mkdir -p "$(dirname "$dst")"
  local ssh_cmd
  ssh_cmd="$(_ssh_rsync_cmd)"
  if [[ -n "${SKYNET_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      sshpass -p "$SKYNET_PASS" \
      "$RSYNC_BIN" -avh --progress --partial --inplace "$@" \
      -e "$ssh_cmd" \
      "${REMOTE_USER_HOST}:${src}" "$dst"
  else
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      "$RSYNC_BIN" -avh --progress --partial --inplace "$@" \
      -e "$ssh_cmd" \
      "${REMOTE_USER_HOST}:${src}" "$dst"
  fi
}

# --- 1. checkpoints ---------------------------------------------------------
# remote logs/... -> local checkpoints/... (same convention as pull_teleop_*).
declare -a CKPTS=(
  "logs/RBY1_dp3_pcd1024/dp3_pcd1024_glass_2k/checkpoints/epoch_epoch=1299.ckpt|checkpoints/RBY1_dp3_pcd1024/dp3_pcd1024_glass_2k/checkpoints/epoch_epoch=1299.ckpt"
  "logs/RBY1_dp3_pcd1024/dp3_pcd1024_glass_2k/checkpoints/epoch_epoch=1999.ckpt|checkpoints/RBY1_dp3_pcd1024/dp3_pcd1024_glass_2k/checkpoints/epoch_epoch=1999.ckpt"
  "logs/RBY1_adapt3r_slamrect/adapt3r_slamrect_2k/checkpoints/epoch_epoch=1999.ckpt|checkpoints/RBY1_adapt3r_slamrect/adapt3r_slamrect_2k/checkpoints/epoch_epoch=1999.ckpt"
)
if [[ "$WITH_2048" == "1" ]]; then
  CKPTS+=("logs/RBY1_dp3_pcd/dp3_pcd2048_glass_2k/checkpoints/epoch_epoch=299.ckpt|checkpoints/RBY1_dp3_pcd/dp3_pcd2048_glass_2k/checkpoints/epoch_epoch=299.ckpt")
fi

for pair in "${CKPTS[@]}"; do
  src="${pair%%|*}"; dst="${pair##*|}"
  echo "=== ckpt: ${src} ==="
  run_rsync "${REMOTE_REPO}/${src}" "./${dst}"
done

# --- 2. datasets ------------------------------------------------------------
if [[ "$SKIP_DATASETS" != "1" ]]; then
  declare -a DATASETS=(
    "datasets/rby1_teleop_pcd1024_glass"
    "datasets/rby1_teleop_slamrect_rgbd"
  )
  [[ "$WITH_2048" == "1" ]] && DATASETS+=("datasets/rby1_teleop_pcd2048_glass")
  for ds in "${DATASETS[@]}"; do
    echo "=== dataset: ${ds} ==="
    run_rsync "${REMOTE_REPO}/${ds}/" "./${ds}/"
  done
fi

# --- 3. code snapshot -------------------------------------------------------
if [[ "$SKIP_CODE" != "1" ]]; then
  echo "=== code snapshot: egomimic/ (*.py, *.yaml, *.json) -> skynet_snapshot/ ==="
  run_rsync "${REMOTE_REPO}/egomimic/" "./skynet_snapshot/egomimic/" \
    -m \
    --include='*/' \
    --include='*.py' --include='*.yaml' --include='*.yml' --include='*.json' \
    --exclude='*'
  date -Is > ./skynet_snapshot/PULLED_AT.txt
  echo "Snapshot in ./skynet_snapshot/egomimic — now run:"
  echo "    python apply_skynet_snapshot.py           # dry-run diff report"
  echo "    python apply_skynet_snapshot.py --apply   # copy new+changed files in"
fi

echo ""
echo "Done. Local checkpoints:"
ls -lh checkpoints/RBY1_dp3_pcd1024/dp3_pcd1024_glass_2k/checkpoints/ 2>/dev/null || true
ls -lh checkpoints/RBY1_adapt3r_slamrect/adapt3r_slamrect_2k/checkpoints/ 2>/dev/null || true
[[ "$WITH_2048" == "1" ]] && ls -lh checkpoints/RBY1_dp3_pcd/dp3_pcd2048_glass_2k/checkpoints/ 2>/dev/null || true
echo "Datasets:"
du -sh datasets/rby1_teleop_pcd1024_glass datasets/rby1_teleop_slamrect_rgbd 2>/dev/null || true
