#!/bin/bash
#SBATCH --job-name=convert_mug2_zarr
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2/convert-zarr-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2/convert-zarr-%j.err
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

# real_mug v2 (bimanual FR3 + ORCA, 30 Hz): processed FAIVE H5 (made on Euler by
# eth_clariden/mug/calib/raw_to_processed_v2.py) -> LeRobot (--arm both, base_frame,
# CL 45 -> 100) -> zarr (eve_bimanual: front + left/right wrist cams).
# Submit with `sbatch -A <account> ...` to override the account above.

set -eo pipefail
ulimit -c 0

EGOMAIN=${EGOMAIN:-/iopsstor/scratch/cscs/jiaqchen/code/EgoVerse-main}
export EGOMAIN
FAIVE2LEROBOT=${FAIVE2LEROBOT:-/iopsstor/scratch/cscs/jiaqchen/code/faive2lerobot}
PROCESSED=${PROCESSED:-/iopsstor/scratch/cscs/jiaqchen/data/real_mug_v2/processed}
LEROBOT_OUT=${LEROBOT_OUT:-/iopsstor/scratch/cscs/jiaqchen/data/real_mug_v2/lerobot}
ZARR_OUT=${ZARR_OUT:-/iopsstor/scratch/cscs/jiaqchen/data/real_mug_v2/zarr}
LEROBOT_NAME=${LEROBOT_NAME:-mug2_lerobot_base_frame_30hz_both}
mkdir -p "$LEROBOT_OUT" "$ZARR_OUT" /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2
[ "$(ls $PROCESSED/*.h5 2>/dev/null | wc -l)" -gt 0 ] || { echo "no processed files in $PROCESSED"; exit 1; }

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml bash -c '
set -eo pipefail
EGOMAIN='"${EGOMAIN}"'; export EGOMAIN
FAIVE2LEROBOT='"${FAIVE2LEROBOT}"'
PROCESSED='"${PROCESSED}"'
LEROBOT_OUT='"${LEROBOT_OUT}"'
ZARR_OUT='"${ZARR_OUT}"'
LEROBOT_NAME='"${LEROBOT_NAME}"'
source ${EGOMAIN}/eth_clariden/clariden.sh
LEROBOT_DIR="${LEROBOT_OUT}/${LEROBOT_NAME}"
echo "[mug2 conversion] both arms, base_frame, 30 Hz, top + left/right wrist"
echo "  PROCESSED=${PROCESSED} ($(ls ${PROCESSED}/*.h5 | wc -l) files)  LEROBOT_DIR=${LEROBOT_DIR}  ZARR_OUT=${ZARR_OUT}"

if [ -f "${LEROBOT_DIR}/meta/info.json" ]; then
    echo "[step 1] LeRobot dataset already exists at ${LEROBOT_DIR}; skipping"
else
    echo "[step 1] processed H5 -> LeRobot (faive2lerobot, --arm both, 30 Hz)"
    cd ${FAIVE2LEROBOT}
    python convert_faive_to_lerobot.py \
        --name ${LEROBOT_NAME} \
        --arm both \
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

if compgen -G "${ZARR_OUT}/episode_*.zarr" > /dev/null; then
    echo "[step 2] zarr episodes already exist at ${ZARR_OUT}; skipping"
else
    echo "[step 2] LeRobot -> zarr (eve_bimanual, top + both wrists)"
    python ${EGOMAIN}/egomimic/scripts/eve_process/lerobot_to_zarr.py \
        --root ${LEROBOT_DIR} \
        --output-dir ${ZARR_OUT} \
        --embodiment eve_bimanual \
        --fps 30 \
        --task-name real_mug_v2 \
        --task-description "Bimanual FR3+ORCA cup-on-saucer handover (cvg-dex-teleop v2, base_frame per arm, 30 Hz, CL 45->100)"
fi
echo "=== zarr episodes: $(ls -d ${ZARR_OUT}/episode_*.zarr 2>/dev/null | wc -l)"; du -sh ${LEROBOT_DIR} ${ZARR_OUT} 2>/dev/null
'
