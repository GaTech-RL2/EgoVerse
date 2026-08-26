#!/usr/bin/env bash
set -euo pipefail

# 2026-08-18 session — the three FRAME-TRANSFORMED human policies.
# (see ai_docs/hw_session_0818_guide.md)
#
#   dp3_hglass   human DP3, glass frame, rect-Z crop   val 0.0825
#   dp3_eefball  human DP3, 1.5 m ball around right eef val 0.0828
#   a3r_eef      LUT RGB + depth + per-frame eef extrinsic val 0.0840
#
# All three beat the camera-frame a3r_human (0.0903); h_rect (0.0651) is still
# the RGB bar.
#
# ⚠ THE CODE SYNC IS MANDATORY, not optional. Verified 2026-08-18: this
# checkout has ZERO `eef_T` references while the remote's hpt.py has them, and
# commit 13fe12f7 tracks the previously-untracked encoder file. Old checkouts
# cannot unpickle these checkpoints, and a3r_eef's serving guard (which raises
# on a missing eef_T rather than silently running the wrong frame) is absent.
#
# We do NOT `git pull` here: this checkout carries 79 local changes, 6 of them
# on the exact tracked files the new commits touch. Instead we rsync the remote
# WORKING TREE into skynet_snapshot/ and let apply_skynet_snapshot.py stage the
# diff for review — the same flow used for every prior session.
#
# Usage:  cd ~/RB_Y1_workspace/EgoVerse && bash pull_hw_0818.sh
#           SKIP_DATASETS=1 bash pull_hw_0818.sh    # ckpts + code only
#           LIST_ONLY=1     bash pull_hw_0818.sh

REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky2.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
SKIP_DATASETS="${SKIP_DATASETS:-0}"
LIST_ONLY="${LIST_ONLY:-0}"

RSYNC_BIN="/usr/bin/rsync"; [[ -x "$RSYNC_BIN" ]] || RSYNC_BIN="$(command -v rsync)"
SKYNET_KEY="${SKYNET_KEY:-$HOME/.ssh/id_ed25519_czhang_skynet}"
SSH_STR="ssh -i ${SKYNET_KEY} -o IdentitiesOnly=yes -o PreferredAuthentications=publickey -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

_rsync() {
  local src="$1" dst="$2"; shift 2
  mkdir -p "$(dirname "$dst")"
  env -u LD_LIBRARY_PATH -u CONDA_PREFIX \
    "$RSYNC_BIN" -avh --progress --partial --inplace "$@" -e "$SSH_STR" \
    "${REMOTE_USER_HOST}:${src}" "$dst"
}

# run_rel | local_rel | port | label
CKPTS=(
"logs/RBY1_dp3_human_glass/dp3_human_glass_2k/checkpoints/epoch_epoch=1999.ckpt|checkpoints/RBY1_dp3_human_glass/dp3_human_glass_2k/checkpoints/epoch_epoch=1999.ckpt|8004|dp3_hglass"
"logs/RBY1_dp3_eefball/dp3_eefball_2k/checkpoints/epoch_epoch=1999.ckpt|checkpoints/RBY1_dp3_eefball/dp3_eefball_2k/checkpoints/epoch_epoch=1999.ckpt|8005|dp3_eefball"
"logs/RBY1_adapt3r_human_eef/adapt3r_human_eef_2k/checkpoints/epoch_epoch=1999.ckpt|checkpoints/RBY1_adapt3r_human_eef/adapt3r_human_eef_2k/checkpoints/epoch_epoch=1999.ckpt|8006|a3r_eef"
)
# Only the two DP3 replay sets by default: human_fullpp_rgbd_eef is 14 GB and is
# needed only to replay a3r_eef locally.
DATASETS=(human_dp3_robotglass human_dp3_eefball)

if [[ "$LIST_ONLY" == "1" ]]; then
  for row in "${CKPTS[@]}"; do echo "  ${row%%|*}"; done; exit 0
fi

echo "=== 1. Final checkpoints (ep1999 each) ==="
for row in "${CKPTS[@]}"; do
  IFS='|' read -r rel local port label <<< "$row"
  echo "  --- ${label}  -> port ${port}"
  _rsync "${REMOTE_REPO}/${rel}" "./${local}"
done

echo ""
echo "=== 2. Code snapshot (MANDATORY — eef_T routing + tracked encoder) ==="
_rsync "${REMOTE_REPO}/egomimic/" "./skynet_snapshot/egomimic/" \
  -m --include='*/' --include='*.py' --include='*.yaml' --include='*.yml' \
  --include='*.json' --exclude='*'
date -Is > ./skynet_snapshot/PULLED_AT.txt

echo ""
echo "=== 3. Docs + LUT assets (dryref_new3.txt = reference MAEs) ==="
_rsync "${REMOTE_REPO}/ai_docs/hw_session_0818_guide.md" "./ai_docs/"
_rsync "${REMOTE_REPO}/ai_docs/assets_rect_lut/" "./ai_docs/assets_rect_lut/"

if [[ "$SKIP_DATASETS" != "1" ]]; then
  echo ""
  echo "=== 4. Replay datasets (838M each) ==="
  for ds in "${DATASETS[@]}"; do
    echo "  --- datasets/${ds}"
    _rsync "${REMOTE_REPO}/datasets/${ds}/" "./datasets/${ds}/"
  done
fi

echo ""
echo "=== Done ==="
for row in "${CKPTS[@]}"; do
  IFS='|' read -r rel local port label <<< "$row"
  [[ -f "$local" ]] && printf "  %-14s %6s  %s\n" "$label" \
    "$(du -h "$local" | cut -f1)" "$(date -r "$local" +%Y-%m-%d_%H:%M)"
done
df -h / | tail -1 | awk '{print "  disk free: "$4}'
echo ""
echo "NEXT (mandatory before serving these ckpts):"
echo "  python apply_skynet_snapshot.py            # review the eef_T code diff"
echo "  python apply_skynet_snapshot.py --apply"
