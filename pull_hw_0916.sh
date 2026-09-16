#!/usr/bin/env bash
# Pull the 0916 8-policy fleet (4 basket + 4 pp) + score tables + example npz.
# basket_youngwoong / basket_pooled still training (finish ~04-05:00 09-17): re-run for finals.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H=${SSH_HOST:-sky2-czhang}
B=/coc/flash7/czhang883/Documents/EgoVerse/logs
D="$HERE/checkpoints/hw0916"; mkdir -p "$D"
pull(){ for t in 1 2 3; do rsync -aP -e 'ssh -o ServerAliveInterval=30' "$H:$1" "$2" && return 0; sleep 3; done; return 1; }
pull "$B/RBY1_dp3c_basket_v2/basket60_v2/checkpoints/last.ckpt"      "$D/basket_chuye.ckpt"
pull "$B/RBY1_dp3c_basket/basket60_r1/checkpoints/last.ckpt"         "$D/basket_chuye_overtrimmed.ckpt"
pull "$B/RBY1_dp3c_yw/yw_r1/checkpoints/last.ckpt"                   "$D/basket_youngwoong.ckpt"
pull "$B/RBY1_dp3c_basketmix/basketmix_r1/checkpoints/last.ckpt"     "$D/basket_pooled.ckpt"
pull "$B/RBY1_dp3cmix_v9/mix150_v2_r10_dualeef/checkpoints/last.ckpt" "$D/pp_v9_left.ckpt"
pull "$B/RBY1_dp3cmix_v11/mix150_v2_drop04/checkpoints/last.ckpt"    "$D/pp_v11_drop04.ckpt"
pull "$B/RBY1_dp3cmix_v8/mix150_v2_r10/checkpoints/last.ckpt"        "$D/pp_v8_base.ckpt"
pull "$B/RBY1_dp3cmix_v10/mix150_v2_auxeef/checkpoints/last.ckpt"    "$D/pp_v10_aux.ckpt"
A=/coc/flash7/czhang883/Documents/EgoVerse/ai_docs/assets_rect_lut
for f in dp3cmix_example_1024.npz dryref_mix8911.txt dryref_mix.txt; do pull "$A/$f" "$HERE/ai_docs/assets_rect_lut/$f"; done
ls -la "$D"; echo PULL_0916_DONE
