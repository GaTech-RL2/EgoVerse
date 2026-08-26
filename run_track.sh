#!/bin/bash
# Closed-loop object tracking at 1.5x: sweep the gain on 40 episodes to pick a
# value, using interp resampling (the better of the two open-loop modes).
JOB=${1:?jobid}
R=/coc/flash7/paphiwetsa3/projects/_wt_stack
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/python
SRC=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/circle_4500_plus_gen_v2
LOG=/coc/flash7/scratch/paphiwetsa3/respeed
TMP=/coc/flash7/scratch/paphiwetsa3/respeed/tmp_sweep
mkdir -p "$LOG" "$TMP"
cd "$R" || exit 1

for G in 0.0 0.3 0.6 1.0; do
  rm -rf "$TMP/g$G"
  PYTHONPATH=. SDL_VIDEODRIVER=dummy \
  srun --overlap --jobid="$JOB" "$PY" -m Tsimulation.sim_v2.collect.respeed_dataset \
      --src "$SRC" --dst "$TMP/g$G" \
      --speed-factor 1.5 --limit 40 --sim-version v2 --resample interp \
      --track-gain "$G" \
      --json-out "$LOG/track_g${G}_40.json" 2>&1 \
    | grep -E "^(mean peak|SR@|converted)" | sed "s/^/  gain=$G  /"
done
