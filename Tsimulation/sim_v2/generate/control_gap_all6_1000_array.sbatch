#!/bin/bash
# Generate exactly 1,000 episodes for every fixed
# (controller mode, embodiment) cell: 6 modes x 13 embodiments = 78 tasks.
# Collected demonstrations count toward each cell's 1,000-episode target.
# Each task owns one output directory, so the array is safely resumable.
#SBATCH --job-name=ps_gap6x13_1k
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
# The rl2-lab partition rejects requests above four hours. Slow cells are
# resumed by re-submitting the same array; from_collected never overwrites an
# existing episode and stops only when that cell reaches exactly 1,000.
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --array=0-77%78
#SBATCH --open-mode=append
#SBATCH --output=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826/logs/%x_%A_%a.out

set -euo pipefail

EXP=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826
DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/control_gap_all6x13_1000_per_cell_simv2_20260826
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse-refactor/.venv/bin/python
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun

modes=(ideal tight loose laggy sticky jittery)
agents=(circle circle_small stick L u_socket gripper chain_gripper suction triangle umi scoop flipper spring)

task=${SLURM_ARRAY_TASK_ID:?array task id is required}
mode=${modes[$((task / ${#agents[@]}))]}
agent=${agents[$((task % ${#agents[@]}))]}
source_dir="$EXP/seeds/$mode/$agent/T"
output_root="$DATA/$mode"

# One confirmed collector-idle outlier was quarantined from Ideal/UMI, leaving
# nine clean demonstrations there. Every other cell must contain exactly ten.
expected_sources=10
if [[ "$mode" == ideal && "$agent" == umi ]]; then
  expected_sources=9
fi
source_count=$(find "$source_dir" -maxdepth 1 -type d -name '*.zarr' | wc -l)
if (( source_count != expected_sources )); then
  echo "FATAL: $mode/$agent has $source_count sources; expected $expected_sources" >&2
  exit 2
fi

echo "START task=$task mode=$mode agent=$agent sources=$source_count target=1000"
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
  --target-total 1000 \
  --attempt-batch 32 \
  --seed "$((2608260000 + task * 100000))" \
  --image-size 96

echo "DONE task=$task mode=$mode agent=$agent DATE=$(date --iso-8601=seconds)"
