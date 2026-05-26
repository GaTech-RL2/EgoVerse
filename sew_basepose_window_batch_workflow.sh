#!/bin/bash
# Dispatcher for WINDOWED base_pose-augmented hierarchical training on push_cart.
#
# Per (dataset x variant):
#   1. Step 2b: build <dataset>_basepose_window by adding obs.base_pose_integrated
#      with --window-size 6 --window-delta 10 (= 18-D body-frame past offsets).
#   2. Train using the push_cart_p1_april6d_basepose_window Hydra config.
#
# Assumes the original SEW workflow (sew_hierarchical_batch_workflow.sh) has
# already produced the input LeRobot datasets under ./datasets/<DATASET_NAME>/.
#
# Usage:
#   ./sew_basepose_window_batch_workflow.sh [--dry-run] [--monitor]
#                                           [--datasets ds1,ds2] [--variants v1,v2]
#                                           [--add-variant NAME OVERRIDES]
#
# Naming:
#   logs/<dataset>_basepose_window/<variant>/  (Hydra hydra.run.dir interpolation)

set -e
cd /coc/flash7/zhenyang/EgoVerse

TRAIN_CONFIG="experiments/hierarchical/push_cart_p1_april6d_basepose_window"
WINDOW_SIZE=6
WINDOW_DELTA=10

# Datasets to run: existing LeRobot outputs (Steps 1+2 from the SEW workflow).
DATASETS=(
    "0524_ye_push_cart_new_ours"
    "0524_ye_push_cart_new_mink_eef_only"
)

# Variants: "<name>|<hydra overrides>"
# Default null variant uses config's stem defaults (noise_std=0.10, drop_p=0.0, dim=18).
VARIANTS=(
    "null|"
)

DATASET_FILTER=""
VARIANT_FILTER=""
DRY_RUN=false
MONITOR=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --datasets)  DATASET_FILTER="$2"; shift 2 ;;
        --variants)  VARIANT_FILTER="$2"; shift 2 ;;
        --add-variant) VARIANTS+=("$2|$3"); shift 3 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        --monitor)   MONITOR=true; shift ;;
        -h|--help)   sed -n '1,28p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

in_csv() {
    local needle="$1" csv="$2"
    [ -z "${csv}" ] && return 0
    IFS=',' read -ra parts <<< "${csv}"
    for p in "${parts[@]}"; do
        [ "${p}" = "${needle}" ] && return 0
    done
    return 1
}

JOB_IDS=()
SUBMITTED=()

for DATASET_NAME in "${DATASETS[@]}"; do
    in_csv "${DATASET_NAME}" "${DATASET_FILTER}" || continue

    if [ ! -f "./datasets/${DATASET_NAME}/meta/info.json" ]; then
        echo "ERROR: ./datasets/${DATASET_NAME}/meta/info.json missing." >&2
        echo "       Run sew_hierarchical_batch_workflow.sh on this dataset first." >&2
        exit 2
    fi

    for V_ROW in "${VARIANTS[@]}"; do
        IFS='|' read -r VARIANT_NAME EXTRA_OVERRIDES <<< "${V_ROW}"
        in_csv "${VARIANT_NAME}" "${VARIANT_FILTER}" || continue

        JOB_TAG="sewbpw_${DATASET_NAME}_${VARIANT_NAME}"
        echo "Submitting ${DATASET_NAME} / variant=${VARIANT_NAME}"
        echo "  config=${TRAIN_CONFIG}"
        echo "  window=${WINDOW_SIZE} delta=${WINDOW_DELTA}"
        echo "  overrides=${EXTRA_OVERRIDES:-<none>}"

        if [ "${DRY_RUN}" = true ]; then
            echo "  [dry-run] would sbatch with job-name=${JOB_TAG}"
            continue
        fi

        SUBMIT_OUT=$(sbatch --parsable --job-name="${JOB_TAG}" \
            --export=ALL,DATASET_NAME="${DATASET_NAME}",TRAIN_CONFIG="${TRAIN_CONFIG}",DESCRIPTION="${VARIANT_NAME}",WINDOW_SIZE="${WINDOW_SIZE}",WINDOW_DELTA="${WINDOW_DELTA}",EXTRA_HYDRA_OVERRIDES="${EXTRA_OVERRIDES}" \
            submit_sew_basepose_window_training.sbatch)
        JOB_IDS+=("${SUBMIT_OUT}")
        SUBMITTED+=("${SUBMIT_OUT} ${DATASET_NAME}/${VARIANT_NAME}")
        echo "  job_id=${SUBMIT_OUT}"
    done
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs:"
for s in "${SUBMITTED[@]}"; do
    echo "  ${s}"
done

if [ "${MONITOR}" = true ] && [ ${#JOB_IDS[@]} -gt 0 ]; then
    squeue -j "$(IFS=,; echo "${JOB_IDS[*]}")"
fi
