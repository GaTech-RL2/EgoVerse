#!/bin/bash
#SBATCH -J dexmimicgen_RBY1
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

set -e
cd /coc/flash7/zhenyang/EgoVerse
source emimic/bin/activate
export TMPDIR=/tmp

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DATASET_NAME=${DATASET_NAME}"
echo "RAW_DATA_PATH=${RAW_DATA_PATH}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
nvidia-smi

if [ -z "${DATASET_NAME}" ] || [ -z "${RAW_DATA_PATH}" ]; then
    echo "ERROR: DATASET_NAME and RAW_DATA_PATH env vars must be set"
    exit 1
fi

FPS=${FPS:-10}
TRAIN_CONFIG=${TRAIN_CONFIG:-experiments/dexmimicgen/train_dexmimicgen_no_mobile_joint_act32}
TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-${DATASET_NAME}}
TRAIN_DATASET=${TRAIN_DATASET:-${DATASET_NAME}_lerobot/${DATASET_NAME}}
CONVERSION_CONFIG=${CONVERSION_CONFIG:-./egomimic/rldb/configs/RBY1_dexmimicgen_no_mobile_HDF5_config.json}

LEROBOT_PARENT=./datasets/${DATASET_NAME}_lerobot
LEROBOT_INNER=${LEROBOT_PARENT}/${DATASET_NAME}
LOCK_DIR=./datasets/.${DATASET_NAME}_lerobot.lock

EXPECTED_EPS=$(python -c "import h5py; print(len(h5py.File('${RAW_DATA_PATH}','r')['data'].keys()))")
echo "Expected episodes (from HDF5): ${EXPECTED_EPS}"

read_eps() {
    python -c "
import json
try:
    print(json.load(open('$1'))['total_episodes'])
except Exception:
    print(-1)
"
}

while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
    echo "[Step 1] waiting for conversion lock: ${LOCK_DIR}"
    sleep 30
done
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

NEED_STEP1=1
if [ -f "${LEROBOT_INNER}/meta/info.json" ]; then
    PREV_EPS=$(read_eps "${LEROBOT_INNER}/meta/info.json")
    if [ "${PREV_EPS}" = "${EXPECTED_EPS}" ]; then
        echo "[Step 1] skip, ${LEROBOT_INNER} already has ${PREV_EPS}/${EXPECTED_EPS} episodes"
        NEED_STEP1=0
    else
        echo "[Step 1] partial output detected (${PREV_EPS}/${EXPECTED_EPS}); wiping and redoing"
        rm -rf "${LEROBOT_PARENT}"
    fi
fi

if [ "${NEED_STEP1}" = "1" ]; then
    echo "[Step 1] Converting DexMimicGen HDF5 -> LeRobot"
    rm -rf "${LEROBOT_PARENT}"
    python egomimic/rldb/scripts/robomimic_hd5.py \
        --name "${DATASET_NAME}" \
        --raw-path "${RAW_DATA_PATH}" \
        --dataset-repo-id "${DATASET_NAME}" \
        --config-path "${CONVERSION_CONFIG}" \
        --output-dir "${LEROBOT_PARENT}" \
        --fps ${FPS} \
        --robot-type rby1 \
        --ignore_episode_keys
fi

rmdir "${LOCK_DIR}" 2>/dev/null || true
trap - EXIT

RAW_EPS=$(read_eps "${LEROBOT_INNER}/meta/info.json")
echo "[Sanity] LeRobot episodes: ${RAW_EPS} / expected: ${EXPECTED_EPS}"
if [ "${RAW_EPS}" != "${EXPECTED_EPS}" ]; then
    echo "ERROR: conversion episode count mismatch (${RAW_EPS} vs ${EXPECTED_EPS})"
    exit 2
fi

echo "[Step 2] No egoengine_process step needed; training uses raw actions.joint"
echo "[Step 3] Training with config=${TRAIN_CONFIG}, dataset=datasets/${TRAIN_DATASET}"
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
    --config-name="${TRAIN_CONFIG}" \
    name="${TRAIN_RUN_NAME}" \
    data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${TRAIN_DATASET} \
    data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${TRAIN_DATASET}
