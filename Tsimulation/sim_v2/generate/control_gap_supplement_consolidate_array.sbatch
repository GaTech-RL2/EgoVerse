#!/bin/bash
# Array bounds are supplied by sbatch from the unfinished-primary manifest.
#SBATCH --job-name=ps_gapcon
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --open-mode=append
#SBATCH --output=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826/logs_consolidate/%x_%A_%a.out

set -euo pipefail
EXP=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse-refactor/.venv/bin/python
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun
task=${SLURM_ARRAY_TASK_ID:?array task id is required}
cd "$EXP/source"
export PYTHONPATH="$EXP/source"
"$SRUN" "$PY" -u -m Tsimulation.sim_v2.generate.consolidate_control_gap_supplements \
  --task "$task" --unfinished "$EXP/manifests/unfinished.tsv" \
  --seeds "$EXP/seeds" --shards "$EXP/shards" \
  --supplements "$EXP/supplements"
