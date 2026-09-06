#!/bin/bash
# Submit the tempo-ablation cells: A0 arcvel, A1 arclogdur, A2 arclogdur_ps × {narrow, full} × {42, 43}.
# usage: submit_ablation.sh [variants] [spreads] [seeds]
set -euo pipefail
WT=${WT:-$HOME/scratch/EgoVerse/.claude/worktrees/arc-e1}
OUT=${OUT:-$HOME/scratch/runs/e1_abl}
VARIANTS=${1:-"arcvel arclogdur arclogdur_ps"}; SPREADS=${2:-"narrow full"}; SEEDS=${3:-"42 43"}
mkdir -p $OUT
cd $WT
for v in $VARIANTS; do for s in $SPREADS; do for seed in $SEEDS; do
  jid=$(sbatch --parsable --job-name=e1abl_${v}_${s}_s${seed} --export=ALL,VARIANT=$v,SPREAD=$s,SEED=$seed,OUT=$OUT scripts/e1/fold_ablation.sbatch)
  echo "$(date +%FT%T) $jid $v $s s$seed $(git rev-parse --short HEAD)" | tee -a $OUT/submissions.txt
done; done; done
