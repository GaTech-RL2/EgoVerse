#!/usr/bin/env bash
set -euo pipefail

# Pull everything for the 2026-08-05 hardware session:
#   h_rect / a3r_tel_colour_rgb / a3r_human / dp3_tight1024
# (see ~/Downloads/hw_session_0805_guide.md)
#
# Pulls: 4 x last.ckpt, the rect224 LUT asset, 3 datasets, the deployment
# notes, and a refreshed egomimic code snapshot.
#
# last.ckpt files are ALWAYS re-synced (never skipped): #2 is mid-training and
# #1/#3 finish ep1999 tonight, so re-run this script to pick up newer weights.
# The script prints size+mtime of each so you can record which epoch you got.
#
# Usage:
#   cd ~/RB_Y1_workspace/EgoVerse
#   read -rs -p "Skynet pw: " SKYNET_PASS; export SKYNET_PASS; echo
#   bash pull_hw_0805.sh
#
#   LIST_ONLY=1     bash pull_hw_0805.sh   # show remote sizes/dates, pull nothing
#   SKIP_DATASETS=1 bash pull_hw_0805.sh   # ckpts + LUT + docs only (fast path)
#   CKPTS_ONLY=1    bash pull_hw_0805.sh   # just refresh the 4 last.ckpt

REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky2.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-22}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"
LIST_ONLY="${LIST_ONLY:-0}"
SKIP_DATASETS="${SKIP_DATASETS:-0}"
CKPTS_ONLY="${CKPTS_ONLY:-0}"

RSYNC_BIN="/usr/bin/rsync"; [[ -x "$RSYNC_BIN" ]] || RSYNC_BIN="$(command -v rsync)"

# --- auth: prefer the installed SSH key, fall back to password ---------------
# Keepalives are always on: the 2026-08-04 pull died mid-dataset with
# "Connection closed by <host>" and rsync error 255.
SKYNET_KEY="${SKYNET_KEY:-$HOME/.ssh/id_ed25519_czhang_skynet}"
KEEPALIVE=(-o ServerAliveInterval=30 -o ServerAliveCountMax=6)
if [[ -f "$SKYNET_KEY" ]]; then
  AUTH_OPTS=(-i "$SKYNET_KEY" -o IdentitiesOnly=yes
             -o PreferredAuthentications=publickey)
  AUTH_MODE="key ($SKYNET_KEY)"
else
  AUTH_OPTS=(-o PreferredAuthentications=password -o PubkeyAuthentication=no
             -o IdentitiesOnly=yes)
  AUTH_MODE="password"
fi
SSH_OPTS=(-p "${REMOTE_SSH_PORT}" "${AUTH_OPTS[@]}" "${KEEPALIVE[@]}")
SSH_CMD_STR="ssh -p ${REMOTE_SSH_PORT} ${AUTH_OPTS[*]} ${KEEPALIVE[*]} ${SSH_EXTRA_OPTS}"
echo "[auth] using ${AUTH_MODE}"

_ssh() {
  if [[ "$AUTH_MODE" == "password" && -n "${SKYNET_PASS:-}" ]] \
     && command -v sshpass >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      sshpass -p "$SKYNET_PASS" ssh "${SSH_OPTS[@]}" ${SSH_EXTRA_OPTS} "$REMOTE_USER_HOST" "$@"
  else
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      ssh "${SSH_OPTS[@]}" ${SSH_EXTRA_OPTS} "$REMOTE_USER_HOST" "$@"
  fi
}

_rsync() {
  local src="$1" dst="$2"; shift 2
  mkdir -p "$(dirname "$dst")"
  if [[ "$AUTH_MODE" == "password" && -n "${SKYNET_PASS:-}" ]] \
     && command -v sshpass >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      sshpass -p "$SKYNET_PASS" "$RSYNC_BIN" -avh --progress --partial --inplace "$@" \
      -e "$SSH_CMD_STR" "${REMOTE_USER_HOST}:${src}" "$dst"
  else
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      "$RSYNC_BIN" -avh --progress --partial --inplace "$@" \
      -e "$SSH_CMD_STR" "${REMOTE_USER_HOST}:${src}" "$dst"
  fi
}

# run_rel|local_rel|port|label
CKPTS=(
  "logs/RBY1_human_rect/human_rect_resnet_2k/checkpoints/last.ckpt|checkpoints/RBY1_human_rect/human_rect_resnet_2k/checkpoints/last.ckpt|8000|1 h_rect"
  "logs/RBY1_adapt3r_tel_colour/adapt3r_tel_colour_rgb_2k/checkpoints/last.ckpt|checkpoints/RBY1_adapt3r_tel_colour/adapt3r_tel_colour_rgb_2k/checkpoints/last.ckpt|8001|2 a3r_tel_colour_rgb"
  "logs/RBY1_adapt3r_human/adapt3r_human_2k/checkpoints/last.ckpt|checkpoints/RBY1_adapt3r_human/adapt3r_human_2k/checkpoints/last.ckpt|8002|3 a3r_human"
  "logs/RBY1_dp3_tight1024/dp3_tight1024_2k/checkpoints/last.ckpt|checkpoints/RBY1_dp3_tight1024/dp3_tight1024_2k/checkpoints/last.ckpt|8003|4 dp3_tight1024"
)
DATASETS=(human_fullpp_rgbd rby1_teleop_colour_rgbd rby1_teleop_pcd1024_tight)
DOCS=(ai_docs/human_rect_deployment_note.md ai_docs/rgbd_data_handoff.md
      ai_docs/hw_session_0805_guide.md ai_docs/pcd_policy_deployment_guide.md)
LUT="ai_docs/assets_rect_lut/robot_rect224_lut.npz"

echo "=== Remote inventory (size / mtime) ==="
{
  for row in "${CKPTS[@]}"; do echo "${row%%|*}"; done
  echo "$LUT"
  for d in "${DOCS[@]}"; do echo "$d"; done
} | while IFS= read -r rel; do
  echo "ls -lh --time-style=+%Y-%m-%d_%H:%M ${REMOTE_REPO}/${rel} 2>/dev/null || echo 'MISSING ${rel}'"
done > /tmp/_0805_probe.sh
_ssh "bash -s" < /tmp/_0805_probe.sh | awk '{if ($1=="MISSING") print "  !! " $2; else print "  " $5 "  " $6 "  " $7}'
if [[ "$SKIP_DATASETS" != "1" && "$CKPTS_ONLY" != "1" ]]; then
  echo "  --- dataset sizes ---"
  _ssh "cd ${REMOTE_REPO} && du -sh $(printf 'datasets/%s ' "${DATASETS[@]}") 2>/dev/null" \
    | sed 's/^/  /' || true
fi

if [[ "$LIST_ONLY" == "1" ]]; then echo "(LIST_ONLY=1 — nothing pulled)"; exit 0; fi

echo ""
echo "=== 1. Checkpoints (always re-synced; mid-training weights change) ==="
for row in "${CKPTS[@]}"; do
  IFS='|' read -r rel local port label <<< "$row"
  echo "  --- #${label}  (serve on port ${port}) ---"
  _rsync "${REMOTE_REPO}/${rel}" "./${local}"
done

if [[ "$CKPTS_ONLY" == "1" ]]; then echo "(CKPTS_ONLY=1 — done)"; exit 0; fi

echo ""
echo "=== 2. rect224 LUT asset (REQUIRED by policies 1/2/3) ==="
_rsync "${REMOTE_REPO}/${LUT}" "./${LUT}"

echo ""
echo "=== 3. Deployment notes ==="
for d in "${DOCS[@]}"; do
  _rsync "${REMOTE_REPO}/${d}" "./${d}" || echo "  (skip: ${d} not on remote)"
done

echo ""
echo "=== 4. Refreshed egomimic code snapshot ==="
_rsync "${REMOTE_REPO}/egomimic/" "./skynet_snapshot/egomimic/" \
  -m --include='*/' --include='*.py' --include='*.yaml' --include='*.yml' \
  --include='*.json' --exclude='*'
date -Is > ./skynet_snapshot/PULLED_AT.txt

if [[ "$SKIP_DATASETS" != "1" ]]; then
  echo ""
  echo "=== 5. Datasets (dry-run replay + dataset_avg reset) ==="
  for ds in "${DATASETS[@]}"; do
    echo "  --- datasets/${ds} ---"
    _rsync "${REMOTE_REPO}/datasets/${ds}/" "./datasets/${ds}/"
  done
fi

echo ""
echo "=== Done. Record these (checkpoint identity for attribution): ==="
for row in "${CKPTS[@]}"; do
  IFS='|' read -r rel local port label <<< "$row"
  if [[ -f "$local" ]]; then
    printf "  #%-24s %s  %s\n" "$label" \
      "$(du -h "$local" | cut -f1)" "$(date -r "$local" +%Y-%m-%d_%H:%M)"
  fi
done
ls -lh "$LUT" 2>/dev/null | awk '{print "  LUT: " $5 "  " $9}'
df -h / | tail -1 | awk '{print "  disk free: " $4}'
echo ""
echo "Next: python apply_skynet_snapshot.py           # review code diffs"
echo "      python apply_skynet_snapshot.py --apply   # then serve"
