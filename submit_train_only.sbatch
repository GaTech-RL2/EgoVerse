#!/bin/bash
#SBATCH -J train_only_RBY1
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=32G
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -o ./logs/slurm/%A/out.log
#SBATCH -e ./logs/slurm/%A/err.log
#SBATCH --exclude='puma,deebot,qt-1'

# Train-only job: assumes the LeRobot final dataset already exists at ./datasets/${TRAIN_DATASET}.
#
# Required env vars (via sbatch --export=ALL,...):
#   TRAIN_CONFIG       Hydra config-name (e.g. experiments/hierarchical/push_cart_p1_masked_attn)
#   TRAIN_RUN_NAME     run name for logs/wandb
#   TRAIN_DATASET      LeRobot final dataset folder (under ./datasets/)

set -e
cd /coc/flash7/zhenyang/EgoVerse
source emimic/bin/activate
export TMPDIR=/tmp

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "TRAIN_RUN_NAME=${TRAIN_RUN_NAME}"
echo "TRAIN_DATASET=${TRAIN_DATASET}"
nvidia-smi

if [ -z "${TRAIN_CONFIG}" ] || [ -z "${TRAIN_RUN_NAME}" ] || [ -z "${TRAIN_DATASET}" ]; then
    echo "ERROR: TRAIN_CONFIG, TRAIN_RUN_NAME, TRAIN_DATASET must be set"
    exit 1
fi

FINAL_DIR=/coc/flash7/zhenyang/EgoVerse/datasets/${TRAIN_DATASET}
if [ ! -f "${FINAL_DIR}/meta/info.json" ]; then
    echo "ERROR: dataset not ready at ${FINAL_DIR} (missing meta/info.json)"
    exit 2
fi

echo "[Train-only] Training with config=${TRAIN_CONFIG} on ${FINAL_DIR}"
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
    --config-name="${TRAIN_CONFIG}" \
    name="${TRAIN_RUN_NAME}" \
    data.train_datasets.dataset1.datasets.rl2_lab.folder_path=${FINAL_DIR} \
    data.valid_datasets.dataset1.datasets.eth_lab.folder_path=${FINAL_DIR}
