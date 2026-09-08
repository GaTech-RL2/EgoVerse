#!/usr/bin/env bash
# Pull the dp3cmix fleet (latest last.ckpt, updated in place on sky2) + example obs.
# Re-run after training finishes (~22:00 2026-09-08) to get the finals.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H=${SSH_HOST:-sky2-czhang}
B=/coc/flash7/czhang883/Documents/EgoVerse
D="$HERE/checkpoints/dp3cmix"; mkdir -p "$D"
for v in v1:mix150_base v2:mix150_drop07 v3:mix150_pts2048 v4:mix150_h128; do
  n=${v%%:*}; run=${v##*:}
  rsync -aP -e 'ssh -o ServerAliveInterval=30' \
    "$H:$B/logs/RBY1_dp3cmix_$n/$run/checkpoints/last.ckpt" "$D/dp3cmix_${n}_last.ckpt"
done
rsync -aP "$H:$B/ai_docs/assets_rect_lut/dp3cmix_example_1024.npz" "$HERE/ai_docs/assets_rect_lut/"
rsync -aP "$H:$B/ai_docs/assets_rect_lut/dp3cmix_example_2048.npz" "$HERE/ai_docs/assets_rect_lut/"
ls -la "$D" "$HERE/ai_docs/assets_rect_lut/" | grep -E "dp3cmix|total"
echo PULL_DP3CMIX_DONE
