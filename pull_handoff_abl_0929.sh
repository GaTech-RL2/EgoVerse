#!/bin/bash
# Pull the 2026-09-29 handoff: 39 ckpts (adds W1 W2 W3 GZ), new testdata
# (W_2048, W3_2048LR, GZ_gazecrop), gaze predictor + videos. rsync = only new/changed.
set -u
SRC=sky2-czhang:/coc/flash7/czhang883/handoff_abl
DST=/home/aloha/RB_Y1_workspace/EgoVerse/checkpoints/abl0921
mkdir -p "$DST/testdata" "$DST/gaze"
rsync -a "$SRC/ckpts/" "$DST/" 2>&1 | grep -vE '^\s*\*|^$'
rsync -a "$SRC/testdata/" "$DST/testdata/" 2>&1 | grep -vE '^\s*\*|^$'
rsync -a "$SRC/gaze_predictor.pt" "$SRC/gaze_videos" "$DST/gaze/" 2>&1 | grep -vE '^\s*\*|^$'
echo "--- ckpts: $(ls "$DST"/*.ckpt | wc -l)  testdata: $(ls "$DST"/testdata | wc -l)  gaze: $(ls "$DST"/gaze)"
echo PULL_0929_DONE
