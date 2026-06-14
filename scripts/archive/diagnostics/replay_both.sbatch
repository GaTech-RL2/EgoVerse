#!/bin/bash
#SBATCH --job-name=replay_datasets
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/replay_output_%j.log
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/replay_output_%j.log
#SBATCH --partition=overcap
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=overcap

source /coc/flash7/paphiwetsa3/projects/EgoVerse2/.venv/bin/activate

export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

PYTHON=/coc/flash7/paphiwetsa3/projects/EgoVerse2/.venv/bin/python3.11
SCRIPT=/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/replay_dataset.py
OUT_BASE=/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/replay_videos

echo "=== Replaying new_circle_clean (491 eps, 50 sample) ==="
$PYTHON $SCRIPT /coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_clean $OUT_BASE/new_circle_clean --n_episodes 50

echo ""
echo "=== Replaying new_circle_2 (501 eps, 50 sample) ==="
$PYTHON $SCRIPT /coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_2/new_circle_2 $OUT_BASE/new_circle_2 --n_episodes 50

echo ""
echo "=== DONE ==="
