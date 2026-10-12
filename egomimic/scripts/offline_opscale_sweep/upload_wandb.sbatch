#!/bin/bash
#SBATCH --account=loaner
#SBATCH --partition=rl2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu21
#SBATCH --chdir=/workspace/users/ani-cheluva/EgoVerse
#SBATCH --output=/workspace/users/ani-cheluva/EgoVerse/logs/fixed_budget_opscale_hist/offline_sweep_20261010/wandb-%j.out
set -euo pipefail
source emimic/bin/activate
python -u egomimic/scripts/offline_opscale_sweep/upload_wandb.py "$1"
