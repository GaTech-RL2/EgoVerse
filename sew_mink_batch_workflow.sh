#!/bin/bash
# Batch dispatcher: for each (CSV folder, solver) pair, submit ONE SLURM job that
# runs CSV -> HDF5 (SEW or Mink) -> LeRobot -> training.
#
# Usage:
#   ./sew_mink_batch_workflow.sh                    # uses DEFAULT_INPUTS below
#   ./sew_mink_batch_workflow.sh <csv_folder> ...   # one or more CSV folders
#
# Each input CSV folder produces TWO jobs (one per solver).

set -e
cd /coc/flash7/zhenyang/EgoVerse

DATASET_PREFIX="RBY1"
TRAIN_CONFIG="train_rby1_sew_human_mobie_img_act32_proprio_dropout"
FPS=10
BLACK_IMAGE=true
DELTA_ACTIONS=false
SOLVERS=("sew" "mink")

DEFAULT_INPUTS=(
    /coc/flash7/zhenyang/datasets/ye_basket_table_0518/0518
    /coc/flash7/zhenyang/datasets/ye_push_cart_0518/0518
)

INPUTS=("$@")
if [ ${#INPUTS[@]} -eq 0 ]; then
    INPUTS=("${DEFAULT_INPUTS[@]}")
fi

if [ ${#INPUTS[@]} -eq 0 ]; then
    echo "ERROR: No CSV folders provided. Pass paths as args or edit DEFAULT_INPUTS."
    exit 1
fi

JOB_IDS=()
for CSV_FOLDER in "${INPUTS[@]}"; do
    if [ ! -d "${CSV_FOLDER}" ]; then
        echo "WARN: ${CSV_FOLDER} is not a directory; skipping"
        continue
    fi
    # Build a tag from the parent folder name (e.g. ye_basket_table_0518 / 0518).
    PARENT_NAME=$(basename "$(dirname "${CSV_FOLDER}")")
    CHILD_NAME=$(basename "${CSV_FOLDER}")
    TAG="${PARENT_NAME}"
    if [ "${CHILD_NAME}" != "${PARENT_NAME}" ]; then
        TAG="${PARENT_NAME}_${CHILD_NAME}"
    fi
    SAFE_TAG="${TAG//[^A-Za-z0-9._-]/_}"

    for SOLVER in "${SOLVERS[@]}"; do
        DATASET_NAME="${DATASET_PREFIX}_${SAFE_TAG}_${SOLVER}"
        TRAIN_RUN_NAME="${DATASET_NAME}"

        echo "Submitting ${CSV_FOLDER} [${SOLVER}] -> ${DATASET_NAME}"
        SUBMIT_OUT=$(sbatch --parsable --job-name="${SOLVER}_${SAFE_TAG}" \
            --export=ALL,SOLVER="${SOLVER}",CSV_FOLDER="${CSV_FOLDER}",DATASET_NAME="${DATASET_NAME}",TRAIN_CONFIG="${TRAIN_CONFIG}",FPS="${FPS}",BLACK_IMAGE="${BLACK_IMAGE}",DELTA_ACTIONS="${DELTA_ACTIONS}",TRAIN_RUN_NAME="${TRAIN_RUN_NAME}" \
            submit_csv_sew_mink_train.sbatch)
        JOB_IDS+=("${SUBMIT_OUT}")
        echo "  job_id=${SUBMIT_OUT}"
    done
done

echo ""
echo "Submitted ${#JOB_IDS[@]} job(s): ${JOB_IDS[*]}"
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    squeue -j "$(IFS=,; echo "${JOB_IDS[*]}")" 2>&1 || true
fi
