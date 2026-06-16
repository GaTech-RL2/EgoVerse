#!/bin/bash
#SBATCH --job-name=robotwin_ricl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=egomimic/ricl/outputs/robotwin_train/slurm_%j.out
#SBATCH --error=egomimic/ricl/outputs/robotwin_train/slurm_%j.err

# Train EgoVerse pi0.5 RICL on RoboTwin (single task, in-memory HDF5 shim).
# Submit from the repo root: sbatch egomimic/ricl/scripts/train_robotwin_ricl.sbatch
set -uo pipefail
cd /coc/flash7/rco3/EgoVerse
mkdir -p egomimic/ricl/outputs/robotwin_train
source emimic/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ROOT="${ROOT:-egomimic/ricl/outputs/robotwin_raw/extracted}"

python -u egomimic/ricl/scripts/train_robotwin_ricl.py --stage full \
  --root "${ROOT}" \
  --embed "${EMBED:-dinov2}" \
  --k "${K:-4}" \
  --action-horizon "${ACTION_HORIZON:-15}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --num-workers "${NUM_WORKERS:-6}" \
  --devices 1 \
  --max-steps "${MAX_STEPS:-4000}" \
  --warmup-steps "${WARMUP_STEPS:-200}" \
  --lr "${LR:-3e-5}" \
  --adam8bit \
  --val-every "${VAL_EVERY:-250}" \
  --limit-val-batches "${LIMIT_VAL:-10}" \
  --ckpt-every "${CKPT_EVERY:-1000}" \
  --out egomimic/ricl/outputs/robotwin_train \
  --name "${RUN_NAME:-robotwin_ricl}"

echo "=== train exit: $? ==="
