#!/bin/bash
#SBATCH --job-name=convert_mug_zarr
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug/convert-raw-zarr-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug/convert-raw-zarr-%j.err
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

# real_mug (Franka Panda + ORCA right hand, 30 Hz):
#   raw recorder H5 -> processed FAIVE H5 (raw_to_processed.py: Panda FK proprio, cam renames)
#                   -> LeRobot (faive2lerobot, --arm right, base_frame, 30 Hz: CL=45 -> 100)
#                   -> zarr (eve_right_arm, front + right wrist)
# Submit with `sbatch -A <account> ...` to override the account above.
# Pass --limit N (env LIMIT) for a smoke test on the first N episodes.

set -eo pipefail
ulimit -c 0

EGOMAIN=${EGOMAIN:-/iopsstor/scratch/cscs/jiaqchen/code/EgoVerse-main}
export EGOMAIN
FAIVE2LEROBOT=${FAIVE2LEROBOT:-/iopsstor/scratch/cscs/jiaqchen/code/faive2lerobot}
RAW=${RAW:-/iopsstor/scratch/cscs/jiaqchen/data/real_mug_30hz/raw}
PROCESSED=${PROCESSED:-/iopsstor/scratch/cscs/jiaqchen/data/real_mug_30hz/processed}
LEROBOT_OUT=${LEROBOT_OUT:-/iopsstor/scratch/cscs/jiaqchen/data/real_mug_30hz/lerobot}
ZARR_OUT=${ZARR_OUT:-/iopsstor/scratch/cscs/jiaqchen/data/real_mug_30hz/zarr}
LEROBOT_NAME=${LEROBOT_NAME:-mug_lerobot_base_frame_30hz_right}
LIMIT=${LIMIT:-}

mkdir -p "$PROCESSED" "$LEROBOT_OUT" "$ZARR_OUT" /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml bash -c '
set -eo pipefail
EGOMAIN='"${EGOMAIN}"'; export EGOMAIN
FAIVE2LEROBOT='"${FAIVE2LEROBOT}"'
RAW='"${RAW}"'
PROCESSED='"${PROCESSED}"'
LEROBOT_OUT='"${LEROBOT_OUT}"'
ZARR_OUT='"${ZARR_OUT}"'
LEROBOT_NAME='"${LEROBOT_NAME}"'
LIMIT='"${LIMIT}"'

source ${EGOMAIN}/eth_clariden/clariden.sh
LEROBOT_DIR="${LEROBOT_OUT}/${LEROBOT_NAME}"

echo "============================================================"
echo "[mug conversion] right_arm, base_frame, 30 Hz, front + right wrist"
echo "  RAW=${RAW}  PROCESSED=${PROCESSED}"
echo "  LEROBOT_DIR=${LEROBOT_DIR}  ZARR_OUT=${ZARR_OUT}  LIMIT=${LIMIT:-all}"
echo "============================================================"

##################### STEP 0: raw -> processed FAIVE H5 #####################
echo "[step 0] raw recorder H5 -> processed FAIVE layout"
python ${FAIVE2LEROBOT}/raw_to_processed.py \
    --raw-dir ${RAW} --out-dir ${PROCESSED} --min-frames 60 ${LIMIT:+--limit ${LIMIT}}

##################### STEP 1: processed H5 -> LeRobot #####################
if [ -f "${LEROBOT_DIR}/meta/info.json" ]; then
    echo "[step 1] LeRobot dataset already exists at ${LEROBOT_DIR}; skipping"
else
    echo "[step 1] processed H5 -> LeRobot (faive2lerobot, --arm right, 30 Hz)"
    cd ${FAIVE2LEROBOT}
    # 30 Hz: 45 source steps = 1.5 s horizon (== 75 @ 50 Hz), resampled to 100.
    python convert_faive_to_lerobot.py \
        --name ${LEROBOT_NAME} \
        --arm right \
        --extrinsics-key best_calib \
        --raw-path ${PROCESSED} \
        --dataset-repo-id 0 \
        --fps 30 \
        --prestack \
        --nproc 10 \
        --nthreads 10 \
        --output-dir ${LEROBOT_OUT} \
        --POINT_GAP_ACT 1 \
        --CHUNK_LENGTH_ACT 45 \
        --CHUNK_LENGTH_ACT_OUT 100 \
        --base-frame \
        --hand-action-start-offset 1 \
        --overwrite
fi

##################### STEP 2: LeRobot -> zarr #####################
if compgen -G "${ZARR_OUT}/episode_*.zarr" > /dev/null; then
    echo "[step 2] zarr episodes already exist at ${ZARR_OUT}; skipping"
else
    echo "[step 2] LeRobot -> zarr (eve_right_arm, front + right wrist)"
    python ${EGOMAIN}/egomimic/scripts/eve_process/lerobot_to_zarr.py \
        --root ${LEROBOT_DIR} \
        --output-dir ${ZARR_OUT} \
        --embodiment eve_right_arm \
        --fps 30 \
        --task-name real_mug \
        --task-description "Franka+ORCA right arm real_mug (cvg-dex-teleop recorded, base_frame, 30 Hz, PG=1 CL=45->100)"
fi

echo "=== Final state ==="
ls ${LEROBOT_DIR}/meta/ 2>/dev/null
echo "zarr episodes: $(ls -d ${ZARR_OUT}/episode_*.zarr 2>/dev/null | wc -l)"
du -sh ${PROCESSED} ${LEROBOT_DIR} ${ZARR_OUT} 2>/dev/null
'
