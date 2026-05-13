#!/bin/bash
#SBATCH --job-name=l2z_cup
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/lerobot-to-zarr-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/lerobot-to-zarr-%j.err
#SBATCH --partition=normal
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Standalone lerobot -> zarr conversion. Output cup_full_zarr/ used by all 6
# cup variants (BC + 5 cotrain). Idempotent only by file presence: delete
# cup_full_zarr/ to force a fresh run.

set -eo pipefail
ulimit -c 0

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main

LEROBOT_ROOT=/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/cup_full_lerobot/cup_lerobot_base_frame_pg4
ZARR_OUT=/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/cup_full_zarr
mkdir -p "$(dirname "${ZARR_OUT}")" /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup

echo "[l2z cup] LEROBOT_ROOT=${LEROBOT_ROOT}"
echo "[l2z cup] ZARR_OUT=${ZARR_OUT}"

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml \
     --container-workdir=${EGOMAIN} \
     bash -lc '
set -eo pipefail
EGOMAIN='"${EGOMAIN}"'
LEROBOT_ROOT='"${LEROBOT_ROOT}"'
ZARR_OUT='"${ZARR_OUT}"'

source ${EGOMAIN}/eth_clariden/clariden.sh

python ${EGOMAIN}/egomimic/scripts/eve_process/lerobot_to_zarr.py \
    --root ${LEROBOT_ROOT} \
    --output-dir ${ZARR_OUT} \
    --embodiment eve_bimanual \
    --task-name cup_on_saucer \
    --task-description "EVE bimanual cup_on_saucer (50Hz, PG=4, base_frame, wrist cams)"

echo "=== final zarr count ==="
ls ${ZARR_OUT} | wc -l
du -sh ${ZARR_OUT}
'
