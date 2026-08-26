#!/bin/bash
# Obstacle-gen check: 100 episodes spread evenly across obstacle levels 1-30,
# at 0.5x (best slow variant) and 1.5x interp (best fast variant).
JOB=${1:?jobid}
R=/coc/flash7/paphiwetsa3/projects/_wt_stack
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/python
SRC=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/circle_4500_plus_gen_v2
OUT=/coc/flash7/paphiwetsa3/datasets/Tsim_v2
LOG=/coc/flash7/scratch/paphiwetsa3/respeed
GEN='_obs([1-9]|[12][0-9]|30)_'
cd "$R" || exit 1

run () { # tag  factor  resample
  echo "######## gen $1"
  PYTHONPATH=. SDL_VIDEODRIVER=dummy \
  srun --overlap --jobid="$JOB" "$PY" -m Tsimulation.sim_v2.collect.respeed_dataset \
    --src "$SRC" --dst "$OUT/circle4500gen_v2_pusher${1}_gen100" \
    --speed-factor "$2" --resample "$3" --sim-version v2 \
    --include "$GEN" --sample even --limit 100 \
    --json-out "$LOG/respeed_${1}_gen100.json" 2>&1 \
  | grep -vE "Unstable|v3_unstable" | tail -7
  echo
}

run 0.5x 0.5 hold
run 1.5x 1.5 interp
