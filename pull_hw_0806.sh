#!/usr/bin/env bash
set -euo pipefail

# 2026-08-06 hardware session — pull the FINAL checkpoints by exact epoch name.
# (see ~/Downloads/hw_session_0806_guide.md and a3r_human_realworld_eval_guide.md)
#
# Why not reuse pull_hw_0805.sh: that one pulls `last.ckpt`, which was still
# mid-training on 08-04. Training finished 08-05; the finals have stable names
# (ep1999, except dp3_tight1024 whose run ends at ep1899). Pulling by name makes
# the checkpoint identity unambiguous in the rollout log.
#
# Datasets and the LUT are already local (verified 2026-08-17) -> not re-pulled
# unless WITH_DATASETS=1.
#
# Usage:
#   cd ~/RB_Y1_workspace/EgoVerse && bash pull_hw_0806.sh
#     LIST_ONLY=1      bash pull_hw_0806.sh   # inventory only
#     A3R_HUMAN_ONLY=1 bash pull_hw_0806.sh   # just policy #3 (328 MB)
#     SKIP_CODE=1      bash pull_hw_0806.sh   # no egomimic snapshot refresh
#     WITH_DATASETS=1  bash pull_hw_0806.sh   # re-sync the 3 replay datasets

REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky2.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-22}"
LIST_ONLY="${LIST_ONLY:-0}"
A3R_HUMAN_ONLY="${A3R_HUMAN_ONLY:-0}"
SKIP_CODE="${SKIP_CODE:-0}"
WITH_DATASETS="${WITH_DATASETS:-0}"

RSYNC_BIN="/usr/bin/rsync"; [[ -x "$RSYNC_BIN" ]] || RSYNC_BIN="$(command -v rsync)"

# Key auth confirmed working 2026-08-17 (BatchMode, no password prompt).
SKYNET_KEY="${SKYNET_KEY:-$HOME/.ssh/id_ed25519_czhang_skynet}"
if [[ -f "$SKYNET_KEY" ]]; then
  AUTH=(-i "$SKYNET_KEY" -o IdentitiesOnly=yes -o PreferredAuthentications=publickey)
  echo "[auth] key $SKYNET_KEY"
else
  AUTH=(-o PreferredAuthentications=password -o PubkeyAuthentication=no)
  echo "[auth] password (export SKYNET_PASS and install sshpass)"
fi
# Keepalives: the 08-04 pull died mid-dataset with rsync error 255.
KEEP=(-o ServerAliveInterval=30 -o ServerAliveCountMax=6)
SSH_STR="ssh -p ${REMOTE_SSH_PORT} ${AUTH[*]} ${KEEP[*]}"

_ssh() {
  if [[ -n "${SKYNET_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX sshpass -p "$SKYNET_PASS" \
      ssh -p "$REMOTE_SSH_PORT" "${AUTH[@]}" "${KEEP[@]}" "$REMOTE_USER_HOST" "$@"
  else
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX \
      ssh -p "$REMOTE_SSH_PORT" "${AUTH[@]}" "${KEEP[@]}" "$REMOTE_USER_HOST" "$@"
  fi
}
_rsync() {
  local src="$1" dst="$2"; shift 2
  mkdir -p "$(dirname "$dst")"
  if [[ -n "${SKYNET_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX sshpass -p "$SKYNET_PASS" \
      "$RSYNC_BIN" -avh --progress --partial --inplace "$@" -e "$SSH_STR" \
      "${REMOTE_USER_HOST}:${src}" "$dst"
  else
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX \
      "$RSYNC_BIN" -avh --progress --partial --inplace "$@" -e "$SSH_STR" \
      "${REMOTE_USER_HOST}:${src}" "$dst"
  fi
}

# remote_rel | local_rel | port | label
CKPTS=(
"logs/RBY1_human_rect/human_rect_resnet_2k/checkpoints/epoch_epoch=1999.ckpt|checkpoints/RBY1_human_rect/human_rect_resnet_2k/checkpoints/epoch_epoch=1999.ckpt|8000|1 h_rect"
"logs/RBY1_adapt3r_tel_colour/adapt3r_tel_colour_rgb_2k/checkpoints/epoch_epoch=1999.ckpt|checkpoints/RBY1_adapt3r_tel_colour/adapt3r_tel_colour_rgb_2k/checkpoints/epoch_epoch=1999.ckpt|8001|2 a3r_tel_colour"
"logs/RBY1_adapt3r_human/adapt3r_human_2k/checkpoints/epoch_epoch=1999.ckpt|checkpoints/RBY1_adapt3r_human/adapt3r_human_2k/checkpoints/epoch_epoch=1999.ckpt|8002|3 a3r_human"
"logs/RBY1_dp3_tight1024/dp3_tight1024_2k/checkpoints/epoch_epoch=1899.ckpt|checkpoints/RBY1_dp3_tight1024/dp3_tight1024_2k/checkpoints/epoch_epoch=1899.ckpt|8003|4 dp3_tight1024"
)
if [[ "$A3R_HUMAN_ONLY" == "1" ]]; then
  CKPTS=("${CKPTS[2]}")
fi
DOCS=(ai_docs/hw_session_0806_guide.md ai_docs/human_rect_deployment_note.md
      ai_docs/pcd_tight_deployment_guide.md ai_docs/fpp_deployment_note.md
      ai_docs/rgbd_data_handoff.md)
DATASETS=(human_fullpp_rgbd rby1_teleop_colour_rgbd rby1_teleop_pcd1024_tight)

echo "=== Remote inventory ==="
{ for r in "${CKPTS[@]}"; do echo "${r%%|*}"; done; for d in "${DOCS[@]}"; do echo "$d"; done; } \
  | while IFS= read -r rel; do
      echo "ls -lh --time-style=+%Y-%m-%d_%H:%M '${REMOTE_REPO}/${rel}' 2>/dev/null || echo 'MISSING ${rel}'"
    done > /tmp/_0806_probe.sh
_ssh "bash -s" < /tmp/_0806_probe.sh \
  | awk '{if ($1=="MISSING") print "  !! "$2; else print "  "$5"  "$6"  "$7}'
[[ "$LIST_ONLY" == "1" ]] && { echo "(LIST_ONLY=1 — nothing pulled)"; exit 0; }

echo ""
echo "=== 1. Final checkpoints (by exact epoch name) ==="
for row in "${CKPTS[@]}"; do
  IFS='|' read -r rel local port label <<< "$row"
  echo "  --- #${label}   -> port ${port}"
  _rsync "${REMOTE_REPO}/${rel}" "./${local}"
done

echo ""
echo "=== 2. Deployment notes ==="
for d in "${DOCS[@]}"; do _rsync "${REMOTE_REPO}/${d}" "./${d}" || echo "  (skip ${d})"; done

if [[ "$SKIP_CODE" != "1" ]]; then
  echo ""
  echo "=== 3. egomimic code snapshot refresh ==="
  _rsync "${REMOTE_REPO}/egomimic/" "./skynet_snapshot/egomimic/" \
    -m --include='*/' --include='*.py' --include='*.yaml' --include='*.yml' \
    --include='*.json' --exclude='*'
  date -Is > ./skynet_snapshot/PULLED_AT.txt
fi

if [[ "$WITH_DATASETS" == "1" ]]; then
  echo ""
  echo "=== 4. Replay datasets ==="
  for ds in "${DATASETS[@]}"; do _rsync "${REMOTE_REPO}/datasets/${ds}/" "./datasets/${ds}/"; done
fi

echo ""
echo "=== Done — record these (checkpoint identity for the rollout log) ==="
for row in "${CKPTS[@]}"; do
  IFS='|' read -r rel local port label <<< "$row"
  [[ -f "$local" ]] && printf "  #%-18s %6s  %s  %s\n" "$label" \
    "$(du -h "$local" | cut -f1)" "$(date -r "$local" +%Y-%m-%d_%H:%M)" "$local"
done
df -h / | tail -1 | awk '{print "  disk free: "$4}'
echo ""
echo "Next: python apply_skynet_snapshot.py           # review code diffs"
echo "      python apply_skynet_snapshot.py --apply   # then serve"
