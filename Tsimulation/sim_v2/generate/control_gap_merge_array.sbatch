#!/bin/bash
# Merge completed private shards into one final cell. One task owns each final
# controller/embodiment directory, so merges cannot collide.
#SBATCH --job-name=ps_gapmerge
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-77%78
#SBATCH --open-mode=append
#SBATCH --output=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826/logs_merge/%x_%A_%a.out

set -euo pipefail

EXP=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826
DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/control_gap_all6x13_1000_per_cell_simv2_20260826
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse-refactor/.venv/bin/python
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun

task=${SLURM_ARRAY_TASK_ID:?array task id is required}
echo "START merge task=$task HOST=$(hostname) DATE=$(date --iso-8601=seconds)"

cd "$EXP/source"
export PYTHONPATH="$EXP/source"

"$SRUN" "$PY" -u -m Tsimulation.sim_v2.generate.merge_control_gap_shards \
  --task "$task" \
  --cells "$EXP/manifests/cells.tsv" \
  --seeds "$EXP/seeds" \
  --data "$DATA" \
  --shards "$EXP/shards" \
  --quarantine "$EXP/quarantine_partial_before_4x" \
  --target 1000

echo "DONE merge task=$task DATE=$(date --iso-8601=seconds)"
