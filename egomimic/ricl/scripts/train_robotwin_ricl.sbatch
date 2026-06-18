#!/bin/bash
#SBATCH --job-name=robotwin_ricl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --partition=gpu-h200
#SBATCH --account=gts-dxu345-rl2
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=inferno
#SBATCH --time=04:00:00
#SBATCH --output=egomimic/ricl/outputs/robotwin_train/slurm_%j.out
#SBATCH --error=egomimic/ricl/outputs/robotwin_train/slurm_%j.err

# Train EgoVerse pi0.5 RICL on RoboTwin (single task, in-memory HDF5 shim).
# Submit from the repo root: sbatch egomimic/ricl/scripts/train_robotwin_ricl.sbatch
set -uo pipefail
cd /storage/project/r-dxu345-0/rco3/EgoVerse2
mkdir -p egomimic/ricl/outputs/robotwin_train
source emimic/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ROOT="${ROOT:-egomimic/ricl/outputs/robotwin_raw/extracted}"

python -u egomimic/ricl/scripts/train_robotwin_ricl.py --stage full \
  --root "${ROOT}" \
  ${EVAL_ROOT:+--eval-root "${EVAL_ROOT}"} \
  ${BANK_ROOT:+--bank-root "${BANK_ROOT}"} \
  ${EVAL_BANK_ROOT:+--eval-bank-root "${EVAL_BANK_ROOT}"} \
  --embed "${EMBED:-dinov2}" \
  --k "${K:-4}" \
  --action-horizon "${ACTION_HORIZON:-15}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --num-workers "${NUM_WORKERS:-6}" \
  --devices 1 \
  --max-steps "${MAX_STEPS:-4000}" \
  --warmup-steps "${WARMUP_STEPS:-200}" \
  --lr "${LR:-3e-5}" \
  ${NO_INCONTEXT:+--no-incontext} \
  ${NO_RANDOM:+--no-random} \
  --val-every "${VAL_EVERY:-250}" \
  --limit-val-batches "${LIMIT_VAL:-10}" \
  --ckpt-every "${CKPT_EVERY:-1000}" \
  --out egomimic/ricl/outputs/robotwin_train \
  --name "${RUN_NAME:-robotwin_ricl}"

echo "=== train exit: $? ==="
