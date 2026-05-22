#!/bin/bash
#SBATCH -J sew_mink_RBY1
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

# End-to-end job: raw CSV folder -> HDF5 (SEW or Mink) -> LeRobot raw -> LeRobot final -> train.
#
# Required env vars (set via sbatch --export=ALL,...):
#   SOLVER             "sew" | "mink"
#   CSV_FOLDER         path to a folder of body_pose_*.csv files
#   DATASET_NAME       string used to name LeRobot dirs and train run
#   TRAIN_RUN_NAME     run name passed to trainHydra (defaults to DATASET_NAME)
#   TRAIN_CONFIG       Hydra train config name (default: train_rby1_sew_human_mobie_img_act32_proprio_dropout)
#   HDF5_OUT_DIR       where the CSV->HDF5 step writes (default: <CSV_FOLDER>_converted_<SOLVER>)
#   HDF5_NAME          basename of the produced HDF5 (default: robot_data_<SOLVER>.hdf5)
#   FPS                LeRobot fps (default: 10)
#   BLACK_IMAGE        true | false (default: true)
#   DELTA_ACTIONS      true | false (default: false)

set -e
cd /coc/flash7/zhenyang/EgoVerse
export TMPDIR=/tmp

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SOLVER=${SOLVER}"
echo "CSV_FOLDER=${CSV_FOLDER}"
echo "DATASET_NAME=${DATASET_NAME}"
nvidia-smi

if [ -z "${SOLVER}" ] || [ -z "${CSV_FOLDER}" ] || [ -z "${DATASET_NAME}" ]; then
    echo "ERROR: SOLVER, CSV_FOLDER, and DATASET_NAME must be set"
    exit 1
fi
case "${SOLVER}" in
    sew|mink) ;;
    *) echo "ERROR: SOLVER must be 'sew' or 'mink' (got '${SOLVER}')"; exit 1 ;;
esac

FPS=${FPS:-10}
BLACK_IMAGE=${BLACK_IMAGE:-true}
DELTA_ACTIONS=${DELTA_ACTIONS:-false}
TRAIN_CONFIG=${TRAIN_CONFIG:-train_rby1_sew_human_mobie_img_act32_proprio_dropout}
TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-${DATASET_NAME}}
TRAIN_DATASET=${TRAIN_DATASET:-${DATASET_NAME}_human_data}
HDF5_OUT_DIR=${HDF5_OUT_DIR:-/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET_NAME}_csv_to_hdf5}
HDF5_NAME=${HDF5_NAME:-robot_data_${SOLVER}.hdf5}
RAW_DATA_PATH="${HDF5_OUT_DIR}/${HDF5_NAME}"

RAW_LEROBOT_DIR=./datasets/${DATASET_NAME}_lerobot_raw
RAW_LEROBOT_INNER=${RAW_LEROBOT_DIR}/${DATASET_NAME}_raw
FINAL_DIR=./datasets/${TRAIN_DATASET}

# ============================================================================
# Step 0: CSV folder -> HDF5 (SEW custom kwargs or Mink baseline).
# ============================================================================
N_CSV=$(ls -1 "${CSV_FOLDER}"/*.csv 2>/dev/null | wc -l)
echo "[Step 0] CSV folder ${CSV_FOLDER} contains ${N_CSV} csv files; SOLVER=${SOLVER}"
if [ "${N_CSV}" = "0" ]; then
    echo "ERROR: no CSV files in ${CSV_FOLDER}"
    exit 4
fi

NEED_STEP0=1
if [ -f "${RAW_DATA_PATH}" ]; then
    PREV_DEMOS=$(python -c "import h5py; print(len(h5py.File('${RAW_DATA_PATH}','r')['data'].keys()))" 2>/dev/null || echo -1)
    if [ "${PREV_DEMOS}" = "${N_CSV}" ]; then
        echo "[Step 0] skip, ${RAW_DATA_PATH} already has ${PREV_DEMOS}/${N_CSV} demos"
        NEED_STEP0=0
    else
        echo "[Step 0] partial output detected (${PREV_DEMOS}/${N_CSV}); wiping and redoing"
        rm -f "${RAW_DATA_PATH}"
    fi
fi

if [ "${NEED_STEP0}" = "1" ]; then
    # Run inside the SEW-Geometric-Teleop checkout (its own venv + PYTHONPATH).
    pushd /coc/flash7/zhenyang/SEW-Geometric-Teleop >/dev/null
    source sew/bin/activate
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export PYNPUT_BACKEND=dummy
    export PYNPUT_BACKEND_KEYBOARD=dummy
    mkdir -p "${HDF5_OUT_DIR}"

    if [ "${SOLVER}" = "sew" ]; then
        python /coc/flash7/zhenyang/EgoVerse/egomimic/scripts/sew_process/csv_to_hdf5_sew_custom.py \
            --input_folder "${CSV_FOLDER}" \
            --output_folder "${HDF5_OUT_DIR}" \
            --hdf5_name "${HDF5_NAME}" \
            --mobile_base_config_path /coc/flash7/zhenyang/EgoVerse/configs/sew_solver_mobile_base_user.yaml
    else
        python projects/xhand_teleop/convert_human_csv_to_robot_hdf5_mink.py \
            --input_folder "${CSV_FOLDER}" \
            --output_folder "${HDF5_OUT_DIR}" \
            --hdf5_name "${HDF5_NAME}"
    fi

    deactivate
    unset PYTHONPATH MUJOCO_GL PYOPENGL_PLATFORM PYNPUT_BACKEND PYNPUT_BACKEND_KEYBOARD
    popd >/dev/null
fi

if [ ! -f "${RAW_DATA_PATH}" ]; then
    echo "ERROR: expected HDF5 not found at ${RAW_DATA_PATH}"
    exit 5
fi
EXPECTED_EPS=$(python3 -c "import h5py; print(len(h5py.File('${RAW_DATA_PATH}','r')['data'].keys()))")
echo "[Step 0] done; HDF5 has ${EXPECTED_EPS} demos at ${RAW_DATA_PATH}"

# ============================================================================
# Steps 1-3: HDF5 -> LeRobot -> train (matches submit_sew_training.sbatch).
# ============================================================================
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
        $( [ "${BLACK_IMAGE}" = true ] && echo "--black-image" ) \
        $( [ "${DELTA_ACTIONS}" = true ] && echo "--add-delta-actions" )
fi
FIN_EPS=$(read_eps "${FINAL_DIR}/meta/info.json")
echo "[Sanity] final episodes: ${FIN_EPS} / expected: ${EXPECTED_EPS}"
if [ "${FIN_EPS}" != "${EXPECTED_EPS}" ]; then
    echo "ERROR: Step 2 episode count mismatch (${FIN_EPS} vs ${EXPECTED_EPS})"
    exit 3
fi

echo "[Step 3] Training with config=${TRAIN_CONFIG}, dataset=${TRAIN_DATASET}"
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py \
    --config-name="${TRAIN_CONFIG}" \
    name="${TRAIN_RUN_NAME}" \
    data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${TRAIN_DATASET} \
    data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${TRAIN_DATASET}
