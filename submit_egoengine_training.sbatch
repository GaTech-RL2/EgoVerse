#!/bin/bash
#SBATCH -J egoengine_RBY1
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
echo "DATASET=${DATASET}"
echo "RAW=${RAW}"
nvidia-smi

if [ -z "${DATASET}" ] || [ -z "${RAW}" ]; then
    echo "ERROR: DATASET and RAW env vars must be set"
    exit 1
fi

FPS=10
RAW_LEROBOT_DIR=./datasets/${DATASET}_lerobot_raw
RAW_LEROBOT_INNER=${RAW_LEROBOT_DIR}/${DATASET}_raw
FINAL_DIR=./datasets/${DATASET}
CONFIG=train_rby1_egoengine_eef_hand_img_act32

# Expected episode count = number of demos in source HDF5.
EXPECTED_EPS=$(python -c "import h5py; print(len(h5py.File('${RAW}','r')['data'].keys()))")
echo "Expected episodes (from HDF5): ${EXPECTED_EPS}"

# Helper: read total_episodes from a LeRobot info.json (echoes -1 if missing/malformed).
read_eps() {
    python -c "
import json, sys
try:
    print(json.load(open('$1'))['total_episodes'])
except Exception:
    print(-1)
"
}

# Step 1: HDF5 -> LeRobot raw. Skip only if a previous run produced a *complete* output.
NEED_STEP1=1
if [ -f "${RAW_LEROBOT_INNER}/meta/info.json" ]; then
    PREV_EPS=$(read_eps "${RAW_LEROBOT_INNER}/meta/info.json")
    if [ "${PREV_EPS}" = "${EXPECTED_EPS}" ]; then
        echo "[Step 1] skip, ${RAW_LEROBOT_INNER} already has ${PREV_EPS}/${EXPECTED_EPS} episodes"
        NEED_STEP1=0
    else
        echo "[Step 1] partial output detected (${PREV_EPS}/${EXPECTED_EPS}); wiping and redoing"
        rm -rf "${RAW_LEROBOT_DIR}"
    fi
fi
if [ "${NEED_STEP1}" = "1" ]; then
    echo "[Step 1] Converting HDF5 -> LeRobot raw"
    rm -rf "${RAW_LEROBOT_DIR}"
    python egomimic/rldb/scripts/robomimic_hd5.py \
        --name "${DATASET}_raw" \
        --raw-path "${RAW}" \
        --dataset-repo-id "${DATASET}_raw" \
        --config-path ./egomimic/rldb/configs/RBY1_egoengine_HDF5_config.json \
        --output-dir "${RAW_LEROBOT_DIR}" \
        --fps ${FPS} \
        --robot-type rby1 \
        --ignore_episode_keys
fi

# Sanity check after Step 1.
RAW_EPS=$(read_eps "${RAW_LEROBOT_INNER}/meta/info.json")
echo "[Sanity] raw episodes: ${RAW_EPS} / expected: ${EXPECTED_EPS}"
if [ "${RAW_EPS}" != "${EXPECTED_EPS}" ]; then
    echo "ERROR: Step 1 episode count mismatch (${RAW_EPS} vs ${EXPECTED_EPS})"
    exit 2
fi

# Step 2: combine eef + hand_right_cmd -> actions.right_arm_eef_hand (18-D).
NEED_STEP2=1
if [ -f "${FINAL_DIR}/meta/info.json" ]; then
    PREV_EPS=$(read_eps "${FINAL_DIR}/meta/info.json")
    if [ "${PREV_EPS}" = "${EXPECTED_EPS}" ]; then
        echo "[Step 2] skip, ${FINAL_DIR} already has ${PREV_EPS}/${EXPECTED_EPS} episodes"
        NEED_STEP2=0
    else
        echo "[Step 2] partial output detected (${PREV_EPS}/${EXPECTED_EPS}); wiping and redoing"
        rm -rf "${FINAL_DIR}"
    fi
fi
if [ "${NEED_STEP2}" = "1" ]; then
    echo "[Step 2] Combining actions"
    rm -rf "${FINAL_DIR}"
    python egomimic/scripts/egoengine_process/egoengine_lerobot_combine_action.py \
        "${RAW_LEROBOT_INNER}" \
        --output-path "${FINAL_DIR}"
fi

# Sanity check after Step 2.
FIN_EPS=$(read_eps "${FINAL_DIR}/meta/info.json")
echo "[Sanity] final episodes: ${FIN_EPS} / expected: ${EXPECTED_EPS}"
if [ "${FIN_EPS}" != "${EXPECTED_EPS}" ]; then
    echo "ERROR: Step 2 episode count mismatch (${FIN_EPS} vs ${EXPECTED_EPS})"
    exit 3
fi

# Step 3: train.
echo "[Step 3] Training"
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
    --config-name=${CONFIG} \
    name=${DATASET} \
    data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET} \
    data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET}
