#!/bin/bash
# Dispatcher for the whole-body + REAL image policy.
# Submits ONE SLURM job per (dataset x head-variant). Each job runs Step 1+2 (data)
# then Step 3 (train) via submit_wb_img_training.sbatch.
#
# Usage:
#   ./wb_img_batch_workflow.sh <DATASET_NAME> <RAW_HDF5_PATH> [vanilla|hier|both]
#
# Example (once data arrives):
#   ./wb_img_batch_workflow.sh RBY1_0623_wb_img /coc/flash7/czhang883/datasets/robot_data.hdf5 both
#
# The two head variants share the SAME converted dataset; the sbatch serializes
# conversion via a mkdir lock, so submitting both at once is safe.

set -e
cd /coc/flash7/czhang883/Documents/EgoVerse

DATASET_NAME="$1"
RAW_DATA_PATH="$2"
WHICH="${3:-both}"

if [ -z "${DATASET_NAME}" ] || [ -z "${RAW_DATA_PATH}" ]; then
    echo "Usage: $0 <DATASET_NAME> <RAW_HDF5_PATH> [vanilla|hier|both]"
    exit 1
fi

declare -A CONFIG
CONFIG[vanilla]="experiments/wholebody_image/wb_img_proprio_vanilla"
CONFIG[hier]="experiments/wholebody_image/wb_img_proprio_hier"

case "${WHICH}" in
    vanilla) VARIANTS=(vanilla) ;;
    hier)    VARIANTS=(hier) ;;
    both)    VARIANTS=(vanilla hier) ;;
    *) echo "Unknown variant '${WHICH}' (use vanilla|hier|both)"; exit 1 ;;
esac

for V in "${VARIANTS[@]}"; do
    echo "Submitting ${DATASET_NAME} / ${V} (config=${CONFIG[$V]})"
    sbatch --job-name="wbimg_${DATASET_NAME}_${V}" \
        --export=ALL,DATASET_NAME="${DATASET_NAME}",RAW_DATA_PATH="${RAW_DATA_PATH}",TRAIN_CONFIG="${CONFIG[$V]}",DESCRIPTION="${V}" \
        submit_wb_img_training.sbatch
done
