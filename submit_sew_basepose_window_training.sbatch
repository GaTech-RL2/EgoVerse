#!/bin/bash
#SBATCH -J sew_basepose_window_RBY1
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

# Counterpart of submit_sew_basepose_training.sbatch for the WINDOWED variant.
# Step 2b builds <dataset>_basepose_window via:
#   egoengine_lerobot_extract_base_pose.py --window-size $WINDOW_SIZE --window-delta $WINDOW_DELTA
# Then trains using the *_basepose_window train config.
#
# Required env vars (set via sbatch --export):
#   DATASET_NAME        existing LeRobot dataset name under ./datasets/
#                       (must be the OUTPUT of Step 2 -- i.e. has actions.joint_base_torso_head_arm_hand)
#   TRAIN_CONFIG        Hydra config-name, e.g. experiments/hierarchical/push_cart_p1_april6d_basepose_window
#   DESCRIPTION         Hydra description override (controls log subdir + wandb id suffix)
# Optional:
#   WINDOW_SIZE              default 6
#   WINDOW_DELTA             default 10
#   EXTRA_HYDRA_OVERRIDES    extra space-separated Hydra overrides

set -e
cd /coc/flash7/zhenyang/EgoVerse
source emimic/bin/activate
export TMPDIR=/tmp

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DATASET_NAME=${DATASET_NAME}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "DESCRIPTION=${DESCRIPTION}"
echo "WINDOW_SIZE=${WINDOW_SIZE:-6}"
echo "WINDOW_DELTA=${WINDOW_DELTA:-10}"
echo "EXTRA_HYDRA_OVERRIDES=${EXTRA_HYDRA_OVERRIDES}"
nvidia-smi

if [ -z "${DATASET_NAME}" ] || [ -z "${TRAIN_CONFIG}" ] || [ -z "${DESCRIPTION}" ]; then
    echo "ERROR: DATASET_NAME, TRAIN_CONFIG, DESCRIPTION env vars must be set"
    exit 1
fi

WINDOW_SIZE=${WINDOW_SIZE:-6}
WINDOW_DELTA=${WINDOW_DELTA:-10}
EXTRA_HYDRA_OVERRIDES=${EXTRA_HYDRA_OVERRIDES:-}

SRC_DIR=./datasets/${DATASET_NAME}
BPW_DIR=./datasets/${DATASET_NAME}_basepose_window
LOCK_DIR=./datasets/.${DATASET_NAME}.basepose_window.lock
EXPECTED_DIM=$((WINDOW_SIZE * 3))

read_eps() {
    python -c "
import json
try:
    print(json.load(open('$1'))['total_episodes'])
except Exception:
    print(-1)
"
}

has_window_key() {
    python -c "
import json, sys
try:
    info = json.load(open('$1'))
    feat = info.get('features', {}).get('obs.base_pose_integrated', {})
    sys.exit(0 if list(feat.get('shape', [])) == [${EXPECTED_DIM}] else 1)
except Exception:
    sys.exit(1)
"
}

if [ ! -f "${SRC_DIR}/meta/info.json" ]; then
    echo "ERROR: Source dataset missing at ${SRC_DIR}/meta/info.json"
    echo "       Run sew_hierarchical_batch_workflow.sh first to produce it."
    exit 2
fi
SRC_EPS=$(read_eps "${SRC_DIR}/meta/info.json")
echo "Source ${SRC_DIR} has ${SRC_EPS} episodes."

# Step 2b: serialize across variants sharing the same dataset.
while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
    echo "[lock] waiting for basepose_window lock: ${LOCK_DIR}"
    sleep 30
done
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

NEED_STEP2B=1
if [ -f "${BPW_DIR}/meta/info.json" ]; then
    BPW_EPS=$(read_eps "${BPW_DIR}/meta/info.json")
    if [ "${BPW_EPS}" = "${SRC_EPS}" ] && has_window_key "${BPW_DIR}/meta/info.json"; then
        echo "[Step 2b] skip, ${BPW_DIR} already has ${BPW_EPS}/${SRC_EPS} eps and obs.base_pose_integrated[dim=${EXPECTED_DIM}]"
        NEED_STEP2B=0
    else
        echo "[Step 2b] partial/wrong-dim output detected (${BPW_EPS}/${SRC_EPS}); wiping and redoing"
        rm -rf "${BPW_DIR}"
    fi
fi
if [ "${NEED_STEP2B}" = "1" ]; then
    echo "[Step 2b] Adding obs.base_pose_integrated (window=${WINDOW_SIZE}, delta=${WINDOW_DELTA}) -> ${BPW_DIR}"
    python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_base_pose.py \
        "${SRC_DIR}" \
        --output-path "${BPW_DIR}" \
        --window-size "${WINDOW_SIZE}" \
        --window-delta "${WINDOW_DELTA}" \
        --overwrite
fi

BPW_EPS=$(read_eps "${BPW_DIR}/meta/info.json")
echo "[Sanity] basepose_window dataset episodes: ${BPW_EPS} / source: ${SRC_EPS}"
if [ "${BPW_EPS}" != "${SRC_EPS}" ]; then
    echo "ERROR: Step 2b episode count mismatch (${BPW_EPS} vs ${SRC_EPS})"
    exit 3
fi

# Release lock before training (training can run concurrent across variants).
rmdir "${LOCK_DIR}" 2>/dev/null || true
trap - EXIT

# Step 3: train. Point the data folder_path overrides at the basepose_window dataset.
TRAIN_DATASET_NAME=${DATASET_NAME}_basepose_window
echo "[Step 3] Training config=${TRAIN_CONFIG}, dataset=${TRAIN_DATASET_NAME}, description=${DESCRIPTION}"
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
    --config-name="${TRAIN_CONFIG}" \
    name="${TRAIN_DATASET_NAME}" \
    "description=\"${DESCRIPTION}\"" \
    data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${TRAIN_DATASET_NAME} \
    data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${TRAIN_DATASET_NAME} \
    ${EXTRA_HYDRA_OVERRIDES}
