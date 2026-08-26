#!/bin/bash
# Array bounds are supplied by sbatch from the generated supplements manifest.
#SBATCH --job-name=ps_gap8x
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --open-mode=append
#SBATCH --output=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826/logs_supplements/%x_%A_%a.out

set -euo pipefail
EXP=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse-refactor/.venv/bin/python
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun
task=${SLURM_ARRAY_TASK_ID:?array task id is required}
row=$(sed -n "$((task + 1))p" "$EXP/manifests/supplements.tsv")
IFS=$'\t' read -r manifest_task original_task cell mode agent original_shard part quota source_count generated original_quota <<< "$row"
if [[ -z "$row" || "$manifest_task" != "$task" || "$quota" -le 0 ]]; then
  echo "FATAL: malformed supplement manifest row for task $task: $row" >&2
  exit 2
fi
source_dir="$EXP/seeds/$mode/$agent/T"
output_root="$EXP/supplements/task$original_task/part$part"
target_total=$((source_count + quota))
echo "START task=$task original_task=$original_task mode=$mode agent=$agent part=$part quota=$quota target=$target_total"
cd "$EXP/source"
export PYTHONPATH="$EXP/source" SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$SRUN" "$PY" -u -m Tsimulation.sim_v2.generate.from_collected \
  --source "$source_dir" --out "$output_root" --target-total "$target_total" \
  --attempt-batch 32 --seed "$((2608270000 + task * 100000))" --image-size 96
echo "DONE task=$task original_task=$original_task DATE=$(date --iso-8601=seconds)"
