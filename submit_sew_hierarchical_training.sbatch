#!/bin/bash
#SBATCH -J sew_hier_RBY1
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

# Required env vars (set via sbatch --export):
#   DATASET_NAME        LeRobot dataset name (also the converted folder under datasets/)
#   RAW_DATA_PATH       Path to source HDF5
#   TRAIN_CONFIG        Hydra config-name (e.g. experiments/hierarchical/rotatebox_p1_april6d)
#   DESCRIPTION         Hydra description override (controls log subdir + wandb id suffix)
# Optional:
#   FPS                 default 10
#   EXTRA_HYDRA_OVERRIDES  extra space-separated Hydra overrides appended after the standard ones

set -e
cd /coc/flash7/zhenyang/EgoVerse
source emimic/bin/activate
export TMPDIR=/tmp

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DATASET_NAME=${DATASET_NAME}"
echo "RAW_DATA_PATH=${RAW_DATA_PATH}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "DESCRIPTION=${DESCRIPTION}"
echo "EXTRA_HYDRA_OVERRIDES=${EXTRA_HYDRA_OVERRIDES}"
nvidia-smi

if [ -z "${DATASET_NAME}" ] || [ -z "${RAW_DATA_PATH}" ] || [ -z "${TRAIN_CONFIG}" ] || [ -z "${DESCRIPTION}" ]; then
    echo "ERROR: DATASET_NAME, RAW_DATA_PATH, TRAIN_CONFIG, DESCRIPTION env vars must be set"
    exit 1
fi

FPS=${FPS:-10}
EXTRA_HYDRA_OVERRIDES=${EXTRA_HYDRA_OVERRIDES:-}

RAW_LEROBOT_DIR=./datasets/${DATASET_NAME}_lerobot_raw
RAW_LEROBOT_INNER=${RAW_LEROBOT_DIR}/${DATASET_NAME}_raw
FINAL_DIR=./datasets/${DATASET_NAME}
LOCK_DIR=./datasets/.${DATASET_NAME}.convert.lock

read_eps() {
    python -c "
import json
try:
    print(json.load(open('$1'))['total_episodes'])
except Exception:
    print(-1)
"
}

if [ -f "${RAW_DATA_PATH}" ]; then
    EXPECTED_EPS=$(python -c "import h5py; print(len(h5py.File('${RAW_DATA_PATH}','r')['data'].keys()))")
    echo "Expected episodes (from HDF5): ${EXPECTED_EPS}"
elif [ -f "${FINAL_DIR}/meta/info.json" ]; then
    EXPECTED_EPS=$(read_eps "${FINAL_DIR}/meta/info.json")
    echo "HDF5 missing at ${RAW_DATA_PATH}; using existing LeRobot at ${FINAL_DIR} (${EXPECTED_EPS} eps). Skipping conversion."
    SKIP_CONVERSION=1
else
    echo "ERROR: HDF5 missing (${RAW_DATA_PATH}) and no existing LeRobot output at ${FINAL_DIR}"
    exit 1
fi

if [ "${SKIP_CONVERSION:-0}" = "1" ]; then
    echo "[Step 1+2] Skipped (HDF5 unavailable, reusing existing LeRobot output)."
else

# Serialize conversion across concurrent variants for the same dataset.
while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
    echo "[lock] waiting for conversion lock: ${LOCK_DIR}"
    sleep 30
done
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

# Step 1: HDF5 -> LeRobot raw.
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
    echo "ERROR: Step 1 episode count mismatch (${RAW_EPS} vs ${EXPECTED_EPS})"
    exit 2
fi

# Step 2: Extract arm/hand / derived action keys.
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
    echo "[Step 2] Extracting arm/hand data"
    rm -rf "${FINAL_DIR}"
    python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py \
        "${RAW_LEROBOT_DIR}" \
        --output-path "${FINAL_DIR}" \
        --black-image
fi

FIN_EPS=$(read_eps "${FINAL_DIR}/meta/info.json")
echo "[Sanity] final episodes: ${FIN_EPS} / expected: ${EXPECTED_EPS}"
if [ "${FIN_EPS}" != "${EXPECTED_EPS}" ]; then
    echo "ERROR: Step 2 episode count mismatch (${FIN_EPS} vs ${EXPECTED_EPS})"
    exit 3
fi

# Release the conversion lock before training (training itself can run concurrent).
rmdir "${LOCK_DIR}" 2>/dev/null || true
trap - EXIT

fi  # end SKIP_CONVERSION guard

# Step 3: train.
# Description is passed as a quoted YAML string so a value like `null` stays a string
# rather than being parsed as YAML None (which would break hydra.run.dir interpolation).
echo "[Step 3] Training config=${TRAIN_CONFIG}, dataset=${DATASET_NAME}, description=${DESCRIPTION}"
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
    --config-name="${TRAIN_CONFIG}" \
    name="${DATASET_NAME}" \
    "description=\"${DESCRIPTION}\"" \
    data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET_NAME} \
    data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET_NAME} \
    ${EXTRA_HYDRA_OVERRIDES}
