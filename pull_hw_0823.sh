#!/usr/bin/env bash
set -euo pipefail

# 2026-08-23 session — the dual / colour / transplant fleet.
# (ai_docs/hw_session_0823_guide.md; priorities 1-3 = dp3c_dual, dp3_dual,
#  dp3_transplant — the headline is #2 vs #3, offline-blind, hardware-only.)
#
# Pulls: 8 final ep1999 ckpts (~2 GB), 5 replay datasets (~7.1 GB),
# dryref_0823.txt, docs, and the egomimic code snapshot (MANDATORY: the
# dual/6-D serving contracts + their new DP3 restore-safety fix live in
# commits 306e329b..c41a3628, after our last sync).
#
# Usage:  cd ~/RB_Y1_workspace/EgoVerse && bash pull_hw_0823.sh
#           PRIORITY_ONLY=1  bash pull_hw_0823.sh   # top-5 ckpts, skip ablations
#           SKIP_DATASETS=1  bash pull_hw_0823.sh

REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky2.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
PRIORITY_ONLY="${PRIORITY_ONLY:-0}"
SKIP_DATASETS="${SKIP_DATASETS:-0}"

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

# run_dir | port | label   (all ckpts are checkpoints/epoch_epoch=1999.ckpt)
CKPTS=(
"RBY1_dp3c_dual/dp3c_dual_2k|8010|1 dp3c_dual (colour, offline champion)"
"RBY1_dp3_dual/dp3_dual_2k|8011|2 dp3_dual"
"RBY1_dp3_transplant/dp3_transplant_2k|8012|3 dp3_transplant (robot-arm clouds)"
"RBY1_dp3_full_eefframe/dp3_full_eefframe_2k|8013|4 dp3_full_eefframe"
"RBY1_dp3_eefframe/dp3_eefframe_2k|8014|5 dp3_eefframe"
)
if [[ "$PRIORITY_ONLY" != "1" ]]; then
  CKPTS+=(
"RBY1_dp3_dual_noprop/dp3_dual_noprop_2k|8015|opt dual_noprop"
"RBY1_dp3_dual_pos3/dp3_dual_pos3_2k|8016|opt dual_pos3"
"RBY1_dp3_dual_eefonly/dp3_dual_eefonly_2k|8017|opt dual_eefonly"
  )
fi
DATASETS=(human_dp3_dual human_dp3_transplant human_dp3c_dual
          human_dp3_eefframe human_dp3_full_eefframe)

echo "=== 1. Checkpoints (final ep1999) ==="
for row in "${CKPTS[@]}"; do
  IFS='|' read -r run port label <<< "$row"
  echo "  --- ${label}  -> port ${port}"
  _rsync "${REMOTE_REPO}/logs/${run}/checkpoints/epoch_epoch=1999.ckpt" \
         "./checkpoints/${run}/checkpoints/epoch_epoch=1999.ckpt"
done

echo ""
echo "=== 2. Code snapshot (MANDATORY — dual/6-D contracts + restore fixes) ==="
_rsync "${REMOTE_REPO}/egomimic/" "./skynet_snapshot/egomimic/" \
  -m --include='*/' --include='*.py' --include='*.yaml' --include='*.yml' \
  --include='*.json' --exclude='*'
date -Is > ./skynet_snapshot/PULLED_AT.txt

echo ""
echo "=== 3. dryref + guide ==="
_rsync "${REMOTE_REPO}/ai_docs/assets_rect_lut/dryref_0823.txt" "./ai_docs/assets_rect_lut/"
_rsync "${REMOTE_REPO}/ai_docs/hw_session_0823_guide.md" "./ai_docs/"

if [[ "$SKIP_DATASETS" != "1" ]]; then
  echo ""
  echo "=== 4. Replay datasets (~7.1 GB total) ==="
  for ds in "${DATASETS[@]}"; do
    echo "  --- datasets/${ds}"
    _rsync "${REMOTE_REPO}/datasets/${ds}/" "./datasets/${ds}/"
  done
fi

echo ""
echo "=== Done — record these (ckpt identity for the rollout log) ==="
for row in "${CKPTS[@]}"; do
  IFS='|' read -r run port label <<< "$row"
  f="checkpoints/${run}/checkpoints/epoch_epoch=1999.ckpt"
  [[ -f "$f" ]] && printf "  %-40s %6s  %s\n" "$label" \
    "$(du -h "$f" | cut -f1)" "$(date -r "$f" +%m-%d_%H:%M)"
done
df -h / | tail -1 | awk '{print "  disk free: "$4}'
echo ""
echo "NEXT (mandatory before serving these ckpts):"
echo "  python apply_skynet_snapshot.py            # review dual/6-D code diff"
echo "  python apply_skynet_snapshot.py --apply"
echo "  # then RE-APPLY the local _batch_E getattr fix if upstream still lacks it"
echo "  # (memory: adapt3r-batch-e-restore-bug)"
