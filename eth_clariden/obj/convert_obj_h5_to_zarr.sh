#!/bin/bash
#SBATCH --job-name=convert_obj_h5_zarr
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/obj/convert-h5-zarr-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/obj/convert-h5-zarr-%j.err
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Convert object_in_container H5 -> LeRobot (faive2lerobot, --arm right) -> zarr.
# Mirrors the bag flow but right-arm only (so action_dim=6, joints=17 in the
# resulting LeRobot dataset, matching OLD repo obj_in_container conventions).

set -eo pipefail

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main
FAIVE2LEROBOT=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/faive2lerobot
RAW=/iopsstor/scratch/cscs/agavryus/egoverse_retry/50hz/object_in_container
LEROBOT_OUT=/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/obj_full_lerobot
ZARR_OUT=/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/obj_full_zarr

mkdir -p "$LEROBOT_OUT" "$ZARR_OUT" /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/obj

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml bash -c '
set -eo pipefail
EGOMAIN='"${EGOMAIN}"'
FAIVE2LEROBOT='"${FAIVE2LEROBOT}"'
RAW='"${RAW}"'
LEROBOT_OUT='"${LEROBOT_OUT}"'
ZARR_OUT='"${ZARR_OUT}"'

source ${EGOMAIN}/eth_clariden/clariden.sh

LEROBOT_NAME="obj_lerobot_base_frame_full_pg4_right"
LEROBOT_DIR="${LEROBOT_OUT}/${LEROBOT_NAME}"

echo "============================================================"
echo "[obj conversion] right_arm, base_frame, wrist cameras"
echo "  RAW=${RAW}"
echo "  LEROBOT_DIR=${LEROBOT_DIR}"
echo "  ZARR_OUT=${ZARR_OUT}"
echo "============================================================"

##################### STEP 1: H5 -> LeRobot #####################
if [ -d "${LEROBOT_DIR}/meta" ] && [ -f "${LEROBOT_DIR}/meta/info.json" ]; then
    echo "[step 1] LeRobot dataset already exists at ${LEROBOT_DIR}; skipping"
else
    echo "[step 1] Converting H5 -> LeRobot (faive2lerobot, --arm right)"
    cd ${FAIVE2LEROBOT}
    python convert_faive_to_lerobot.py \
        --name ${LEROBOT_NAME} \
        --arm right \
        --extrinsics-key best_calib \
        --raw-path ${RAW} \
        --dataset-repo-id 0 \
        --fps 50 \
        --prestack \
        --nproc 10 \
        --nthreads 10 \
        --output-dir ${LEROBOT_OUT} \
        --POINT_GAP_ACT 1 \
        --CHUNK_LENGTH_ACT 75 \
        --CHUNK_LENGTH_ACT_OUT 100 \
        --base-frame \
        --hand-action-start-offset 1 \
        --overwrite
fi

##################### STEP 2: LeRobot -> zarr #####################
if compgen -G "${ZARR_OUT}/episode_*.zarr" > /dev/null; then
    echo "[step 2] Zarr episodes already exist at ${ZARR_OUT}; skipping"
else
    echo "[step 2] Converting LeRobot -> zarr (with wrist cameras)"
    python ${EGOMAIN}/egomimic/scripts/eve_process/lerobot_to_zarr.py \
        --root ${LEROBOT_DIR} \
        --output-dir ${ZARR_OUT} \
        --embodiment eve_right_arm \
        --task-name object_in_container \
        --task-description "EVE right_arm object_in_container (FAIVE recorded, base_frame, PG=4)"
fi

echo "=== Final state ==="
ls ${LEROBOT_DIR}/meta/ 2>/dev/null
echo "---"
ls ${ZARR_OUT} | head -5
echo "..."
echo "total episodes: $(ls -d ${ZARR_OUT}/episode_*.zarr 2>/dev/null | wc -l)"
echo "size: $(du -sh ${ZARR_OUT} | awk "{print \$1}")"
'
