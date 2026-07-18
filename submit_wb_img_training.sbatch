#!/bin/bash
#SBATCH -J wb_img_RBY1
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

# Whole-body (49-D) + REAL image policy: data processing (Step 1+2) then training (Step 3),
# in ONE GPU job, from czhang883's checkout + venv.
#
# KEY DIFFERENCES vs submit_sew_hierarchical_training.sbatch:
#   * runs from /coc/flash7/czhang883/Documents/EgoVerse (czhang883 venv), not zhenyang's
#   * Step 1 uses RBY1_wb_img_HDF5_config.json   -> aria_image + proprio, NO april_tag
#     (april_tag/april_tag_xyz can be all-NaN and crash LeRobot compute_stats; we don't use it)
#   * Step 2 runs WITHOUT --black-image          -> the real image is preserved
#
# PREREQUISITE: the source HDF5 MUST contain obs/aria_image (a real camera frame).
#
# Required env vars (set via sbatch --export):
#   DATASET_NAME    LeRobot dataset name (also the converted folder under datasets/)
#   RAW_DATA_PATH   Path to source HDF5 (must contain obs/aria_image)
#   TRAIN_CONFIG    Hydra --config-name, e.g. experiments/wholebody_image/wb_img_proprio_vanilla
#   DESCRIPTION     Hydra description override (controls log subdir + wandb id suffix)
# Optional:
#   FPS                    default 10
#   EXTRA_HYDRA_OVERRIDES  extra space-separated Hydra overrides appended after the standard ones
#   VAL_DATASET_DIR        external validation LeRobot folder (e.g. the teleop set);
#                          defaults to FINAL_DIR (legacy behavior: val = split of train data)

set -e
REPO_DIR=/coc/flash7/czhang883/Documents/EgoVerse
cd "${REPO_DIR}"
source emimic/bin/activate
export TMPDIR=/tmp
# Redirect caches to flash: the NFS home (/nethome) has a tight quota that torch/HF
# caches will blow during training. (Observed: uv torch extract hit the home quota.)
export XDG_CACHE_HOME=/coc/flash7/czhang883/.cache
export HF_HOME=/coc/flash7/czhang883/.cache/huggingface
export UV_CACHE_DIR=/coc/flash7/czhang883/.cache/uv
# Ensure datasets/ exists, else `mkdir <lock>` fails (missing parent) and the
# conversion-lock loop deadlocks forever (observed on a fresh checkout).
mkdir -p "${REPO_DIR}/datasets"

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

RAW_LEROBOT_DIR=${REPO_DIR}/datasets/${DATASET_NAME}_lerobot_raw
RAW_LEROBOT_INNER=${RAW_LEROBOT_DIR}/${DATASET_NAME}_raw
FINAL_DIR=${REPO_DIR}/datasets/${DATASET_NAME}
LOCK_DIR=${REPO_DIR}/datasets/.${DATASET_NAME}.convert.lock

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
    # Fail fast if the HDF5 has no real image (the whole point of this pipeline).
    HAS_IMG=$(python -c "import h5py; f=h5py.File('${RAW_DATA_PATH}','r'); d=f['data']; k=list(d.keys())[0]; print(int('aria_image' in d[k]['obs']))" 2>/dev/null || echo 0)
    if [ "${HAS_IMG}" != "1" ]; then
        echo "ERROR: ${RAW_DATA_PATH} has no obs/aria_image; this pipeline needs a REAL image."
        echo "       (For black-placeholder april_tag training use submit_sew_hierarchical_training.sbatch.)"
        exit 1
    fi
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

# Serialize conversion across concurrent variants (vanilla + hier) for the same dataset.
while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
    echo "[lock] waiting for conversion lock: ${LOCK_DIR}"
    sleep 30
done
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

# Step 1: HDF5 -> LeRobot raw  (SEW_img config -> KEEPS obs.aria_image)
NEED_STEP1=1
if [ -f "${RAW_LEROBOT_INNER}/meta/info.json" ]; then
    PREV_EPS=$(read_eps "${RAW_LEROBOT_INNER}/meta/info.json")
    if [ "${PREV_EPS}" = "${EXPECTED_EPS}" ]; then
        echo "[Step 1] skip, ${RAW_LEROBOT_INNER} already has ${PREV_EPS}/${EXPECTED_EPS} episodes"
        NEED_STEP1=0
    else
        echo "[Step 1] partial output (${PREV_EPS}/${EXPECTED_EPS}); wiping and redoing"
        rm -rf "${RAW_LEROBOT_DIR}"
    fi
fi
if [ "${NEED_STEP1}" = "1" ]; then
    echo "[Step 1] Converting HDF5 -> LeRobot raw (RBY1_wb_img config, no april_tag)"
    rm -rf "${RAW_LEROBOT_DIR}"
    python egomimic/rldb/scripts/robomimic_hd5.py \
        --name "${DATASET_NAME}_raw" \
        --raw-path "${RAW_DATA_PATH}" \
        --dataset-repo-id "${DATASET_NAME}_raw" \
        --config-path ./egomimic/rldb/configs/RBY1_wb_img_HDF5_config.json \
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

# Step 2: derive 49-D action + 22-D no-wheel proprio.  NO --black-image (keep real image).
NEED_STEP2=1
if [ -f "${FINAL_DIR}/meta/info.json" ]; then
    PREV_EPS=$(read_eps "${FINAL_DIR}/meta/info.json")
    if [ "${PREV_EPS}" = "${EXPECTED_EPS}" ]; then
        echo "[Step 2] skip, ${FINAL_DIR} already has ${PREV_EPS}/${EXPECTED_EPS} episodes"
        NEED_STEP2=0
    else
        echo "[Step 2] partial output (${PREV_EPS}/${EXPECTED_EPS}); wiping and redoing"
        rm -rf "${FINAL_DIR}"
    fi
fi
if [ "${NEED_STEP2}" = "1" ]; then
    echo "[Step 2] Extracting arm/hand + 49-D action (real image preserved)"
    rm -rf "${FINAL_DIR}"
    python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py \
        "${RAW_LEROBOT_DIR}" \
        --output-path "${FINAL_DIR}"
fi

FIN_EPS=$(read_eps "${FINAL_DIR}/meta/info.json")
echo "[Sanity] final episodes: ${FIN_EPS} / expected: ${EXPECTED_EPS}"
if [ "${FIN_EPS}" != "${EXPECTED_EPS}" ]; then
    echo "ERROR: Step 2 episode count mismatch (${FIN_EPS} vs ${EXPECTED_EPS})"
    exit 3
fi

rmdir "${LOCK_DIR}" 2>/dev/null || true
trap - EXIT

fi  # end SKIP_CONVERSION guard

# Step 3: train.
# Offline HF hub for training: the DINOv2 encoder loads timm weights from the warm
# cache on flash; without this, huggingface_hub re-resolves over the network and
# fails hard (~30s of retries) on compute nodes with no/flaky egress. Scoped to
# Step 3 only (conversion above may legitimately want the hub on future datasets).
export HF_HUB_OFFLINE=1
# Auto-resume: overcap is preemptible and --requeue restarts this script from the top;
# without ckpt_path the trainer would restart from epoch 0 and overwrite the run.
AUTO_CKPT="${REPO_DIR}/logs/${DATASET_NAME}/${DESCRIPTION}/checkpoints/last.ckpt"
if [ -f "${AUTO_CKPT}" ]; then
    echo "[Step 3] Found ${AUTO_CKPT}; resuming from it"
    EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES} ckpt_path=${AUTO_CKPT}"
fi
echo "[Step 3] Training config=${TRAIN_CONFIG}, dataset=${DATASET_NAME}, description=${DESCRIPTION}"
VAL_DIR="${VAL_DATASET_DIR:-${FINAL_DIR}}"
echo "[Step 3] Validation dataset: ${VAL_DIR}"
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
    --config-name="${TRAIN_CONFIG}" \
    name="${DATASET_NAME}" \
    "description=\"${DESCRIPTION}\"" \
    data.train_datasets.dataset1.datasets.rl2_lab.folder_path="${FINAL_DIR}" \
    data.valid_datasets.dataset1.datasets.eth_lab.folder_path="${VAL_DIR}" \
    ${EXTRA_HYDRA_OVERRIDES}
