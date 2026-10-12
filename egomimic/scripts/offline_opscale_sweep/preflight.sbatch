#!/bin/bash
#SBATCH --account=loaner
#SBATCH --partition=loaner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --chdir=/workspace/users/ani-cheluva/EgoVerse
#SBATCH --output=/workspace/users/ani-cheluva/EgoVerse/logs/fixed_budget_opscale_hist/offline_sweep_20261010/preflight-%j.out

set -euo pipefail
source emimic/bin/activate
export PYTHONPATH=/workspace/users/ani-cheluva/EgoVerse
python -u egomimic/scripts/offline_opscale_sweep/preflight.py
