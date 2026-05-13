#!/bin/bash
# SEW batch dispatcher: submit ONE SLURM job per HDF5.
# Each job runs SEW data processing (Step 1 + Step 2) and then training.
#
# Usage: ./sew_batch_workflow.sh [hdf5_path ...]
# If no args, the default input list below is used.

set -e
cd /coc/flash7/zhenyang/EgoVerse

# -----------------------------
# Key parameters (batch-level)
# -----------------------------
DATASET_PREFIX="RBY1"
TRAIN_CONFIG="train_rby1_sew_human_mobie_img_act32_proprio_dropout" #train_rby1_sew_human_no_mobie_img_act32"
FPS=10
BLACK_IMAGE=true
DELTA_ACTIONS=false

# Optional run-name behavior:
# - true:  run name = dataset name
# - false: run name = dataset name + suffix below
RUN_NAME_EQUALS_DATASET=true
RUN_NAME_SUFFIX="_mink"

DEFAULT_INPUTS=(
    /coc/flash7/zhenyang/EgoVerse/datasets/0505_ye_rotate_box_mink_eefonly.hdf5
    # /coc/flash7/zhenyang/datasets/0423_zhenyang_pushing_cart/0423_v6/robot_data_0423_cart_mobile_fix_aprtag_v6.hdf5
)

INPUTS=("$@")
if [ ${#INPUTS[@]} -eq 0 ]; then
    INPUTS=("${DEFAULT_INPUTS[@]}")
fi

if [ ${#INPUTS[@]} -eq 0 ]; then
    echo "ERROR: No HDF5 inputs provided. Pass paths as args or edit DEFAULT_INPUTS."
    exit 1
fi

for RAW_DATA_PATH in "${INPUTS[@]}"; do
    BASENAME=$(basename "${RAW_DATA_PATH}" .hdf5)
    DATASET_NAME="${DATASET_PREFIX}_${BASENAME}"

    if [ "${RUN_NAME_EQUALS_DATASET}" = true ]; then
        TRAIN_RUN_NAME="${DATASET_NAME}"
    else
        TRAIN_RUN_NAME="${DATASET_NAME}${RUN_NAME_SUFFIX}"
    fi

    echo "Submitting ${RAW_DATA_PATH} -> ${DATASET_NAME}"
    sbatch --job-name="sew_${BASENAME}" \
        --export=ALL,DATASET_NAME="${DATASET_NAME}",RAW_DATA_PATH="${RAW_DATA_PATH}",TRAIN_CONFIG="${TRAIN_CONFIG}",FPS="${FPS}",BLACK_IMAGE="${BLACK_IMAGE}",DELTA_ACTIONS="${DELTA_ACTIONS}",TRAIN_RUN_NAME="${TRAIN_RUN_NAME}" \
        submit_sew_training.sbatch
done
