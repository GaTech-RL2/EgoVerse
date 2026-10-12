#!/bin/bash
#SBATCH --job-name=opscale-download
#SBATCH --account=loaner
#SBATCH --partition=loaner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --mem-per-gpu=180G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/workspace/users/%u/logs/%x-%j.out
#SBATCH --error=/workspace/users/%u/logs/%x-%j.err

# Download the fixed-budget folding_clothes episodes. The loaner partition
# gives a node exclusively, so the request takes all eight GPUs even though
# this job only moves files.
set -euo pipefail
source /workspace/users/ani-cheluva/EgoVerse/emimic/bin/activate
cd /workspace/users/ani-cheluva/EgoVerse
python egomimic/scripts/data_download/fixed_budget_opscale.py download \
  --master "egomimic/hydra_configs/mono_20260929_master (1).json" \
  --manifest egomimic/hydra_configs/experiments/fixed_budget_opscale/manifest.json \
  --dest /workspace/datasets/fixed_budget_opscale \
  --workers 32
