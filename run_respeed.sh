#!/bin/bash
# Pilot: convert the first 100 episodes of circle_4500_plus_gen_v2 to
# 0.5x and 1.5x pusher-speed datasets, actions = achieved pusher pose.
JOB=${1:?jobid}
R=/coc/flash7/paphiwetsa3/projects/_wt_stack
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/python
SRC=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/circle_4500_plus_gen_v2
OUT=/coc/flash7/paphiwetsa3/datasets/Tsim_v2
LOG=/coc/flash7/scratch/paphiwetsa3/respeed
mkdir -p "$LOG"

cd "$R" || exit 1
for F in 0.5 1.5; do
  DST="$OUT/circle4500gen_v2_pusher${F}x_pilot100"
  echo "############ ${F}x -> $DST"
  PYTHONPATH=. SDL_VIDEODRIVER=dummy MUJOCO_GL=egl \
  srun --overlap --jobid="$JOB" "$PY" -m Tsimulation.sim_v2.collect.respeed_dataset \
      --src "$SRC" --dst "$DST" \
      --speed-factor "$F" --limit 100 --sim-version v2 \
      --action-space pusher \
      --json-out "$LOG/respeed_${F}x_pilot100.json" \
      2>&1 | grep -vE "Unstable|v3_unstable_dtype|^\s*$" | tail -25
  echo
done
