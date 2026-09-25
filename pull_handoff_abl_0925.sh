#!/bin/bash
# Pull the 2026-09-25 ablation handoff: 35 checkpoints (the 26 from 09-21 +
# G2 G3 G8 S1 S2 S3 S4 I2 I3, with the J-family refreshed) and the held-out
# VAL test data (one npz per obs-variant, same underlying episode -> same GT).
# Flat into checkpoints/abl0921/ so ABL=<ID> keeps working; testdata/ beside.
# Re-runnable (rsync).
set -u
SRC=sky2-czhang:/coc/flash7/czhang883/handoff_abl
DST=/home/aloha/RB_Y1_workspace/EgoVerse/checkpoints/abl0921
mkdir -p "$DST/testdata"
rsync -aP "$SRC/ckpts/" "$DST/" 2>&1 | grep -E "\.ckpt$|error|failed"
rsync -aP "$SRC/testdata/" "$DST/testdata/" 2>&1 | grep -E "\.npz$|error|failed"
echo "--- pulled:"; ls "$DST"/*.ckpt | wc -l; du -sh "$DST" "$DST/testdata"
echo PULL_HANDOFF_DONE
