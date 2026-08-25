#!/bin/bash
# Regenerate the two completed Flipper control-gap cells from demonstrations
# collected after the solid, rate-limited auxiliary-bar fix.
#SBATCH --job-name=ps_flipfix500
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --array=0-1%2
#SBATCH --open-mode=append
#SBATCH --output=/coc/flash7/paphiwetsa3/experiments/pushshapes_flipperfix_gapgen_20260826/logs/%x_%A_%a.out

set -euo pipefail

EXP=/coc/flash7/paphiwetsa3/experiments/pushshapes_flipperfix_gapgen_20260826
DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/control_gap_ideal_tight_1000_per_emb_20260826
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse-refactor/.venv/bin/python
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun

modes=(ideal tight)
mode=${modes[${SLURM_ARRAY_TASK_ID:?array task id is required}]}
source_dir="$EXP/seeds/$mode/flipper/T"
output_root="$DATA/$mode"

source_count=$(find "$source_dir" -maxdepth 1 -type d -name '*.zarr' | wc -l)
if (( source_count != 10 )); then
  echo "FATAL: $mode/flipper has $source_count sources, expected exactly 10" >&2
  exit 2
fi

echo "START mode=$mode agent=flipper sources=$source_count target=500"
echo "SOURCE=$source_dir"
echo "OUTPUT=$output_root/flipper/T"
echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"

cd "$EXP/source"
export PYTHONPATH="$EXP/source"
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$SRUN" "$PY" -u -m Tsimulation.sim_v2.generate.from_collected \
  --source "$source_dir" \
  --out "$output_root" \
  --target-total 500 \
  --attempt-batch 32 \
  --seed "$((260826500 + SLURM_ARRAY_TASK_ID * 100000))" \
  --image-size 96

echo "DONE mode=$mode agent=flipper DATE=$(date --iso-8601=seconds)"
