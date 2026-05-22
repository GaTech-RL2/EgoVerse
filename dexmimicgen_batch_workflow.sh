#!/bin/bash
# DexMimicGen no-mobile batch dispatcher.
# Submits one SLURM job per HDF5. Each job converts HDF5 -> LeRobot and then trains.
#
# Usage:
#   ./dexmimicgen_batch_workflow.sh [options] [hdf5_path ...]
#
# Options:
#   --train-config CONFIG      Hydra config-name. Default: experiments/dexmimicgen/train_dexmimicgen_no_mobile_joint_act32
#   --conversion-config PATH   robomimic_hd5.py JSON config.
#   --fps FPS                  Dataset FPS. Default: 10
#   --prefix PREFIX            Dataset prefix. Default: RBY1_dexmimicgen_no_mobile
#   --run-suffix SUFFIX        Suffix appended to TRAIN_RUN_NAME (dataset name unchanged).
#                              Use to run multiple configs against the same converted dataset.
#   --monitor                  Print squeue status for submitted jobs after launch.

set -e
cd /coc/flash7/zhenyang/EgoVerse

DATASET_PREFIX="RBY1_dexmimicgen_no_mobile"
TRAIN_CONFIG="experiments/dexmimicgen/train_dexmimicgen_no_mobile_joint_act32"
CONVERSION_CONFIG="./egomimic/rldb/configs/RBY1_dexmimicgen_no_mobile_HDF5_config.json"
FPS=10
RUN_SUFFIX=""
MONITOR=false

DEFAULT_INPUTS=(
    /coc/flash7/zhenyang/datasets/demo_first_100_policy_training/can_sort_mink_eef_100_final.hdf5
)

INPUTS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --train-config)
            TRAIN_CONFIG="$2"
            shift 2
            ;;
        --conversion-config)
            CONVERSION_CONFIG="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --prefix)
            DATASET_PREFIX="$2"
            shift 2
            ;;
        --run-suffix)
            RUN_SUFFIX="$2"
            shift 2
            ;;
        --monitor)
            MONITOR=true
            shift
            ;;
        --help|-h)
            sed -n '1,22p' "$0"
            exit 0
            ;;
        *)
            INPUTS+=("$1")
            shift
            ;;
    esac
done

if [ ${#INPUTS[@]} -eq 0 ]; then
    INPUTS=("${DEFAULT_INPUTS[@]}")
fi

if [ ${#INPUTS[@]} -eq 0 ]; then
    echo "ERROR: No HDF5 inputs provided. Pass paths as args or edit DEFAULT_INPUTS."
    exit 1
fi

JOB_IDS=()
for RAW_DATA_PATH in "${INPUTS[@]}"; do
    BASENAME=$(basename "${RAW_DATA_PATH}" .hdf5)
    SAFE_BASENAME="${BASENAME//[^A-Za-z0-9._-]/_}"
    DATASET_NAME="${DATASET_PREFIX}_${SAFE_BASENAME}"
    if [ -n "${RUN_SUFFIX}" ]; then
        TRAIN_RUN_NAME="${DATASET_NAME}_${RUN_SUFFIX}"
        JOB_TAG="dex_${SAFE_BASENAME}_${RUN_SUFFIX}"
    else
        TRAIN_RUN_NAME="${DATASET_NAME}"
        JOB_TAG="dex_${SAFE_BASENAME}"
    fi

    echo "Submitting ${RAW_DATA_PATH} -> ${TRAIN_RUN_NAME}"
    SUBMIT_OUT=$(sbatch --parsable --job-name="${JOB_TAG}" \
        --export=ALL,DATASET_NAME="${DATASET_NAME}",RAW_DATA_PATH="${RAW_DATA_PATH}",TRAIN_CONFIG="${TRAIN_CONFIG}",FPS="${FPS}",TRAIN_RUN_NAME="${TRAIN_RUN_NAME}",TRAIN_DATASET="${DATASET_NAME}_lerobot/${DATASET_NAME}",CONVERSION_CONFIG="${CONVERSION_CONFIG}" \
        submit_dexmimicgen_training.sbatch)
    JOB_IDS+=("${SUBMIT_OUT}")
    echo "  job_id=${SUBMIT_OUT}"
done

if [ "${MONITOR}" = true ]; then
    echo "Submitted jobs: ${JOB_IDS[*]}"
    squeue -j "$(IFS=,; echo "${JOB_IDS[*]}")"
fi
