#!/bin/bash
# On skynet: stage seed-0 (roll_ep0) rollout video for each cell/level into a flat dir + tar.
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
STAGE=/tmp/genvids_stage
rm -rf "$STAGE"; mkdir -p "$STAGE"
for c in hptC8_nogen hptC8_gen txC8_nogen txC8_gen hnetC8_nogen hnetC8_gen hnetC4_nogen hnetC4_gen; do
  for d in logs/genlvl_vids/$c/lvl*/; do
    [ -d "$d" ] || continue
    lvl=$(echo "$d" | grep -oE "lvl[0-9]+" | grep -oE "[0-9]+")
    v=$(ls "$d"roll_ep0_*.mp4 2>/dev/null | head -1)
    [ -z "$v" ] && v=$(ls "$d"roll_ep*.mp4 2>/dev/null | head -1)   # fallback: any episode
    if [ -n "$v" ]; then printf -v ln "%02d" "$lvl"; cp "$v" "$STAGE/${c}_level_${ln}.mp4"; fi
  done
done
echo "staged $(ls "$STAGE"/*.mp4 2>/dev/null | wc -l) videos"
tar -czf /tmp/genvids.tar.gz -C "$STAGE" .
ls -la /tmp/genvids.tar.gz
