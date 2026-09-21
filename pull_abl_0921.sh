#!/bin/bash
# Pull the 2026-09-21 ablation fleet (V0 control + 25 ablations) from sky2.
# All ~250 MB each; total ~6.5 GB. Re-runnable (rsync skips complete files).
set -u
B=sky2-czhang:/coc/flash7/czhang883/Documents/EgoVerse/logs
DST=/home/aloha/RB_Y1_workspace/EgoVerse/checkpoints/abl0921
mkdir -p "$DST"

rsync -aP "$B/RBY1_dp3c_basket_v2/basket60_v2/checkpoints/last.ckpt" "$DST/V0.ckpt"

for v in G1 G4 G6 G7 H3 H4 H5 J1 J2 J3 J3b J4 J4b J5 \
         A1 A2 A3 A4 A5 B2 C3 C4 D1 D3 E4; do
  rsync -aP "$B/RBY1_abl_$v/abl/checkpoints/last.ckpt" "$DST/$v.ckpt" \
    || echo "PULL FAILED: $v" >&2
done

echo "--- pulled: ---"
du -sm "$DST"/*.ckpt
echo PULL_ABL_DONE
