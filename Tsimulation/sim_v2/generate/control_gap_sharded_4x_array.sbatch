#!/bin/bash
# One single-threaded simulator per manifest row. Four rows share a logical
# controller/embodiment cell, but every row writes into a private shard.
# This uses four CPUs per cell without concurrent writes to one Zarr directory.
#SBATCH --job-name=ps_gap4x
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --array=0-311%312
#SBATCH --open-mode=append
#SBATCH --output=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826/logs_shards/%x_%A_%a.out

set -euo pipefail

EXP=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse-refactor/.venv/bin/python
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun
MANIFEST="$EXP/manifests/shards.tsv"

task=${SLURM_ARRAY_TASK_ID:?array task id is required}
row=$(sed -n "$((task + 1))p" "$MANIFEST")
if [[ -z "$row" ]]; then
  echo "FATAL: no shard manifest row for task $task" >&2
  exit 2
fi
IFS=$'\t' read -r manifest_task cell mode agent shard quota source_count base_count <<< "$row"
if (( manifest_task != task || quota <= 0 )); then
  echo "FATAL: malformed manifest row: $row" >&2
  exit 2
fi

source_dir="$EXP/seeds/$mode/$agent/T"
output_root="$EXP/shards/$mode/$agent/shard$shard"
target_total=$((source_count + quota))

echo "START task=$task cell=$cell mode=$mode agent=$agent shard=$shard quota=$quota base=$base_count target=$target_total"
echo "SOURCE=$source_dir"
echo "OUTPUT=$output_root/$agent/T"
echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"

cd "$EXP/source"
export PYTHONPATH="$EXP/source"
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$SRUN" "$PY" -u -m Tsimulation.sim_v2.generate.from_collected \
  --source "$source_dir" \
  --out "$output_root" \
  --target-total "$target_total" \
  --attempt-batch 32 \
  --seed "$((2608267000 + task * 100000))" \
  --image-size 96

echo "DONE task=$task cell=$cell mode=$mode agent=$agent shard=$shard DATE=$(date --iso-8601=seconds)"
