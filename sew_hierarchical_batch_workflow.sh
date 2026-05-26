#!/bin/bash
# Hierarchical (P1) batch dispatcher for SEW-style RBY1 datasets.
# For each (dataset x variant) pair, submits ONE SLURM job that:
#   1. Converts HDF5 -> LeRobot raw  (skipped if already present)
#   2. Extracts arm/hand action keys (skipped if already present)
#   3. Trains the chosen Hydra config with optional per-variant overrides.
#
# Concurrent variants sharing a dataset are serialized at the conversion step
# via a mkdir lock inside the sbatch (see submit_sew_hierarchical_training.sbatch).
#
# Naming convention (Hydra: `hydra.run.dir: ./logs/${name}/${description}`):
#   logs/<lerobot_dataset_name>/<variant_name>/
#
# To add datasets or variants in the future, edit the DATASETS and VARIANTS
# tables below. To add a one-off variant from the CLI, use --add-variant.
#
# Usage:
#   ./sew_hierarchical_batch_workflow.sh [options]
#
# Options:
#   --datasets ds1,ds2,...        Filter datasets by name (default: all)
#   --variants v1,v2,...          Filter variants by name (default: all)
#   --add-variant NAME OVERRIDES  Append an ad-hoc variant (Hydra overrides string)
#   --dry-run                     Print sbatch commands without submitting
#   --monitor                     Run squeue after submission for the new job ids
#   -h|--help                     Show this header

set -e
cd /coc/flash7/zhenyang/EgoVerse

# -----------------------------------------------------------------------------
# Task -> Hydra config map
#   Unified on the slim april6d config for all tasks: TinyCNN encoder (cheap
#   because aria_image is a 64x64 black dummy), embed=192, 8 trunk blocks,
#   april_tag (6D) proprio. Per-task differences are only the dataset path,
#   passed via CLI override.
# -----------------------------------------------------------------------------
declare -A TASK_CONFIG
TASK_CONFIG[rotatebox]="experiments/hierarchical/rotatebox_p1_april6d"
TASK_CONFIG[push_cart]="experiments/hierarchical/rotatebox_p1_april6d"
TASK_CONFIG[basket]="experiments/hierarchical/rotatebox_p1_april6d"

# -----------------------------------------------------------------------------
# Dataset table:  "<lerobot_name>|<task>|<hdf5_path>"
# Lerobot name follows: <parent_folder>_<subfolder>  (e.g. 0524_zy_rotatebox_all_ours)
# -----------------------------------------------------------------------------
DATASETS=(
    "0524_zy_rotatebox_all_ours|rotatebox|/coc/flash7/zhenyang/datasets/0524_zy_rotatebox_all/ours/robot_data.hdf5"
    "0524_ye_push_cart_new_ours|push_cart|/coc/flash7/zhenyang/datasets/0524_ye_push_cart_new/ours/robot_data.hdf5"
    "0524_ye_basket_table_redo_sew_ours|basket|/coc/flash7/zhenyang/datasets/0524_ye_basket_table_redo_sew/ours/robot_data.hdf5"
    # 2026-05-24 v2 batch: regenerated HDF5s from czhang883. v2 suffix where the
    # LeRobot name would collide with an earlier run, fresh names otherwise.
    "0524_ye_basket_table_redo_sew_ours_v2|basket|/coc/flash7/zhenyang/datasets/0524_ye_basket_table_redo_sew/ours/robot_data.hdf5"
    "0524_ye_basket_table_redo_sew_mink_eef_only|basket|/coc/flash7/zhenyang/datasets/0524_ye_basket_table_redo_sew/mink_eef_only/robot_data.hdf5"
    "0524_ye_push_cart_new_ours_v2|push_cart|/coc/flash7/zhenyang/datasets/0524_ye_push_cart_new/ours/robot_data.hdf5"
    "0524_ye_push_cart_new_mink_eef_only|push_cart|/coc/flash7/zhenyang/datasets/0524_ye_push_cart_new/mink_eef_only/robot_data.hdf5"
    "0524_zy_rotatebox_ours|rotatebox|/coc/flash7/czhang883/scratch/hwabl/0524_zy_rotatebox/ours/robot_data.hdf5"
    "0524_zy_rotatebox_mink_eef_only|rotatebox|/coc/flash7/czhang883/scratch/hwabl/0524_zy_rotatebox/mink_eef_only/robot_data.hdf5"
    "0524_ye_rotate_box_ours|rotatebox|/coc/flash7/zhenyang/datasets/0524_ye_rotate_box/ours/robot_data.hdf5"
    "0524_ye_rotate_box_mink_eef_only|rotatebox|/coc/flash7/zhenyang/datasets/0524_ye_rotate_box/mink_eef_only/robot_data.hdf5"
)

# -----------------------------------------------------------------------------
# Variant table: "<variant_name>|<hydra overrides string>"
# Variant name controls description (and therefore log subdir + wandb id suffix).
# Empty overrides string => use the config's defaults.
# -----------------------------------------------------------------------------
VARIANTS=(
    "null|"
    "batch128|data.train_dataloader_params.dataset1.batch_size=128 data.valid_dataloader_params.dataset1.batch_size=128 data.train_dataloader_params.dataset1.num_workers=16 data.valid_dataloader_params.dataset1.num_workers=16 model.optimizer.lr=4e-4"
    "nokl|"
    "batch128_nokl|data.train_dataloader_params.dataset1.batch_size=128 data.valid_dataloader_params.dataset1.batch_size=128 data.train_dataloader_params.dataset1.num_workers=16 data.valid_dataloader_params.dataset1.num_workers=16 model.optimizer.lr=4e-4"
)

# -----------------------------------------------------------------------------
# Arg parsing
# -----------------------------------------------------------------------------
DATASET_FILTER=""
VARIANT_FILTER=""
DRY_RUN=false
MONITOR=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --datasets)
            DATASET_FILTER="$2"
            shift 2
            ;;
        --variants)
            VARIANT_FILTER="$2"
            shift 2
            ;;
        --add-variant)
            VARIANTS+=("$2|$3")
            shift 3
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --monitor)
            MONITOR=true
            shift
            ;;
        -h|--help)
            sed -n '1,36p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

in_csv() {
    # in_csv NAME CSV  -> returns 0 if NAME is in comma-separated CSV (or CSV is empty)
    local needle="$1"
    local csv="$2"
    [ -z "${csv}" ] && return 0
    IFS=',' read -ra parts <<< "${csv}"
    for p in "${parts[@]}"; do
        [ "${p}" = "${needle}" ] && return 0
    done
    return 1
}

# -----------------------------------------------------------------------------
# Dispatch loop
# -----------------------------------------------------------------------------
JOB_IDS=()
SUBMITTED=()

for DS_ROW in "${DATASETS[@]}"; do
    IFS='|' read -r DATASET_NAME TASK RAW_DATA_PATH <<< "${DS_ROW}"
    in_csv "${DATASET_NAME}" "${DATASET_FILTER}" || continue

    TRAIN_CONFIG="${TASK_CONFIG[${TASK}]:-}"
    if [ -z "${TRAIN_CONFIG}" ]; then
        echo "ERROR: no train config registered for task '${TASK}' (dataset ${DATASET_NAME})" >&2
        exit 1
    fi
    if [ ! -f "${RAW_DATA_PATH}" ]; then
        if [ -f "./datasets/${DATASET_NAME}/meta/info.json" ]; then
            echo "  WARN: HDF5 not found at ${RAW_DATA_PATH}; LeRobot output exists, sbatch will reuse it"
        else
            echo "ERROR: HDF5 not found: ${RAW_DATA_PATH} (and no existing LeRobot output)" >&2
            exit 1
        fi
    fi

    for V_ROW in "${VARIANTS[@]}"; do
        IFS='|' read -r VARIANT_NAME EXTRA_OVERRIDES <<< "${V_ROW}"
        in_csv "${VARIANT_NAME}" "${VARIANT_FILTER}" || continue

        JOB_TAG="sewhier_${DATASET_NAME}_${VARIANT_NAME}"
        echo "Submitting ${DATASET_NAME} / variant=${VARIANT_NAME} / task=${TASK}"
        echo "  config=${TRAIN_CONFIG}"
        echo "  overrides=${EXTRA_OVERRIDES:-<none>}"

        if [ "${DRY_RUN}" = true ]; then
            echo "  [dry-run] would sbatch with job-name=${JOB_TAG}"
            continue
        fi

        SUBMIT_OUT=$(sbatch --parsable --job-name="${JOB_TAG}" \
            --export=ALL,DATASET_NAME="${DATASET_NAME}",RAW_DATA_PATH="${RAW_DATA_PATH}",TRAIN_CONFIG="${TRAIN_CONFIG}",DESCRIPTION="${VARIANT_NAME}",EXTRA_HYDRA_OVERRIDES="${EXTRA_OVERRIDES}" \
            submit_sew_hierarchical_training.sbatch)
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
