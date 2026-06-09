#!/bin/bash
#SBATCH --job-name=droid_ricl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=inferno
#SBATCH -A gts-dxu345-rl2
#SBATCH --time=08:00:00
#SBATCH --output=egomimic/ricl/outputs/droid_train/slurm_%j.out
#SBATCH --error=egomimic/ricl/outputs/droid_train/slurm_%j.err

set -uo pipefail
mkdir -p egomimic/ricl/outputs/droid_train

# EgoVerse env (openpi/flax/torch + pi05 checkpoint); EgoVerse2 ricl code on path
source /storage/project/r-dxu345-0/rco3/EgoVerse/emimic/bin/activate
export PYTHONPATH=/storage/project/r-dxu345-0/rco3/EgoVerse2
cd /storage/project/r-dxu345-0/rco3/EgoVerse2

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u egomimic/ricl/scripts/train_droid_ricl.py --stage full \
  --cache-dir egomimic/ricl/outputs/droid_cache \
  --max-steps "${MAX_STEPS:-4000}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --val-every "${VAL_EVERY:-250}" \
  --limit-val-batches "${LIMIT_VAL:-10}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --out egomimic/ricl/outputs/droid_train \
  --name "${RUN_NAME:-droid_ricl}" \
  ${WANDB:+--wandb --wandb-project "${WANDB_PROJECT:-ricl}"}

echo "=== train exit: $? ==="
