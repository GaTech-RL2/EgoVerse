#!/bin/bash
#SBATCH -J hdf5_to_train_RBY1
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

# HDF5 (already produced) -> LeRobot raw -> LeRobot final -> train.
#
# Required env vars (via sbatch --export=ALL,...):
#   RAW_DATA_PATH      path to existing robomimic HDF5
#   DATASET_NAME       base name (used for LeRobot dirs + run name)
#   TRAIN_CONFIG       Hydra config (default: train_rby1_sew_human_mobie_img_act32_proprio_dropout)
#   FPS                LeRobot fps (default: 10)
#   BLACK_IMAGE        true | false (default: true)
#   DELTA_ACTIONS      true | false (default: false)

set -e
cd /coc/flash7/zhenyang/EgoVerse
export TMPDIR=/tmp

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RAW_DATA_PATH=${RAW_DATA_PATH}"
echo "DATASET_NAME=${DATASET_NAME}"
nvidia-smi

if [ -z "${RAW_DATA_PATH}" ] || [ -z "${DATASET_NAME}" ]; then
    echo "ERROR: RAW_DATA_PATH and DATASET_NAME must be set"
    exit 1
fi
if [ ! -f "${RAW_DATA_PATH}" ]; then
    echo "ERROR: HDF5 not found at ${RAW_DATA_PATH}"
    exit 1
fi

FPS=${FPS:-10}
BLACK_IMAGE=${BLACK_IMAGE:-true}
DELTA_ACTIONS=${DELTA_ACTIONS:-false}
TRAIN_CONFIG=${TRAIN_CONFIG:-train_rby1_sew_human_mobie_img_act32_proprio_dropout}
TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-${DATASET_NAME}}
TRAIN_DATASET=${TRAIN_DATASET:-${DATASET_NAME}_human_data}

RAW_LEROBOT_DIR=./datasets/${DATASET_NAME}_lerobot_raw
RAW_LEROBOT_INNER=${RAW_LEROBOT_DIR}/${DATASET_NAME}_raw
FINAL_DIR=./datasets/${TRAIN_DATASET}

source emimic/bin/activate

read_eps() {
    python -c "
import json
try:
    print(json.load(open('$1'))['total_episodes'])
except Exception:
    print(-1)
"
}

EXPECTED_EPS=$(python3 -c "import h5py; print(len(h5py.File('${RAW_DATA_PATH}','r')['data'].keys()))")
echo "[Source] HDF5 has ${EXPECTED_EPS} demos at ${RAW_DATA_PATH}"

NEED_STEP1=1
if [ -f "${RAW_LEROBOT_INNER}/meta/info.json" ]; then
    PREV_EPS=$(read_eps "${RAW_LEROBOT_INNER}/meta/info.json")
    if [ "${PREV_EPS}" = "${EXPECTED_EPS}" ]; then
        echo "[Step 1] skip, ${RAW_LEROBOT_INNER} already has ${PREV_EPS}/${EXPECTED_EPS} episodes"
        NEED_STEP1=0
    else
        echo "[Step 1] partial output (${PREV_EPS}/${EXPECTED_EPS}); wiping"
        rm -rf "${RAW_LEROBOT_DIR}"
    fi
fi
if [ "${NEED_STEP1}" = "1" ]; then
    echo "[Step 1] HDF5 -> LeRobot raw"
    rm -rf "${RAW_LEROBOT_DIR}"
    python egomimic/rldb/scripts/robomimic_hd5.py \
        --name "${DATASET_NAME}_raw" \
        --raw-path "${RAW_DATA_PATH}" \
        --dataset-repo-id "${DATASET_NAME}_raw" \
        --config-path ./egomimic/rldb/configs/RBY1_SEW_lowdim_HDF5_config.json \
        --output-dir "${RAW_LEROBOT_DIR}" \
        --fps ${FPS} \
        --robot-type rby1 \
        --ignore_episode_keys
fi
RAW_EPS=$(read_eps "${RAW_LEROBOT_INNER}/meta/info.json")
echo "[Sanity] raw episodes: ${RAW_EPS} / expected: ${EXPECTED_EPS}"
if [ "${RAW_EPS}" != "${EXPECTED_EPS}" ]; then
    echo "ERROR: Step 1 mismatch"
    exit 2
fi

NEED_STEP2=1
if [ -f "${FINAL_DIR}/meta/info.json" ]; then
    PREV_EPS=$(read_eps "${FINAL_DIR}/meta/info.json")
    if [ "${PREV_EPS}" = "${EXPECTED_EPS}" ]; then
        echo "[Step 2] skip, ${FINAL_DIR} already has ${PREV_EPS}/${EXPECTED_EPS} episodes"
        NEED_STEP2=0
    else
        echo "[Step 2] partial (${PREV_EPS}/${EXPECTED_EPS}); wiping"
        rm -rf "${FINAL_DIR}"
    fi
fi
if [ "${NEED_STEP2}" = "1" ]; then
    echo "[Step 2] extract arm/hand"
    rm -rf "${FINAL_DIR}"
    python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py \
        "${RAW_LEROBOT_DIR}" \
        --output-path "${FINAL_DIR}" \
        $( [ "${BLACK_IMAGE}" = true ] && echo "--black-image" ) \
        $( [ "${DELTA_ACTIONS}" = true ] && echo "--add-delta-actions" )
fi
FIN_EPS=$(read_eps "${FINAL_DIR}/meta/info.json")
echo "[Sanity] final episodes: ${FIN_EPS} / expected: ${EXPECTED_EPS}"
if [ "${FIN_EPS}" != "${EXPECTED_EPS}" ]; then
    echo "ERROR: Step 2 mismatch"
    exit 3
fi

echo "[Step 3] Training config=${TRAIN_CONFIG} run=${TRAIN_RUN_NAME} dataset=${TRAIN_DATASET}"
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
    --config-name="${TRAIN_CONFIG}" \
    name="${TRAIN_RUN_NAME}" \
    data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${TRAIN_DATASET} \
    data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${TRAIN_DATASET}
