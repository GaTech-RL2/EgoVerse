#!/bin/bash
#SBATCH --job-name=T_cup_BC_DEBUG
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/debug/slurm-cup-BC-debug-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/debug/slurm-cup-BC-debug-%j.err
#SBATCH --partition=normal
#SBATCH --requeue
#SBATCH --signal=USR1@600
##################### SBATCH RESOURCES #####################
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
############################################################
# DEBUG cup BC variant. Inline lerobot->zarr conversion (skipped if exists).
# Verifies wrist cams + GT trajectory projection on validation_video.

set -eo pipefail
ulimit -c 0

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main

export LEROBOT_ROOT=${LEROBOT_ROOT:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/cup_full_lerobot/cup_lerobot_base_frame_pg4}
export ZARR_OUT=${ZARR_OUT:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/cup_full_zarr}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/DEBUG/eve_cup_bc/${SLURM_JOB_ID}}

mkdir -p "${HYDRA_RUN_DIR}" "$(dirname "${ZARR_OUT}")" /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/debug

echo "============================================================"
echo "[cup BC DEBUG] EgoVerse-main BC pipeline w/ wrist cams"
echo "  LEROBOT_ROOT=${LEROBOT_ROOT}"
echo "  ZARR_OUT=${ZARR_OUT}"
echo "  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "============================================================"

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml \
     --container-workdir=${EGOMAIN} \
     bash -lc '
set -eo pipefail
EGOMAIN='"${EGOMAIN}"'
LEROBOT_ROOT='"${LEROBOT_ROOT}"'
ZARR_OUT='"${ZARR_OUT}"'
HYDRA_RUN_DIR='"${HYDRA_RUN_DIR}"'

source ${EGOMAIN}/eth_clariden/clariden.sh
nvidia-smi --query-gpu=name,memory.total --format=csv || true

##################### STEP 1: verify zarr present (built by run_lerobot_to_zarr.sh) #####################
if [ ! -d "${ZARR_OUT}" ] || [ "$(ls ${ZARR_OUT}/*.zarr 2>/dev/null | wc -l)" -lt 50 ]; then
    echo "FATAL: cup_full_zarr missing or too few episodes. Run run_lerobot_to_zarr.sh first."
    exit 1
fi
echo "[step 1] cup_full_zarr OK ($(ls ${ZARR_OUT}/*.zarr 2>/dev/null | wc -l) episodes)"

##################### STEP 2: train (debug, 4 epochs, val every 2) #####################
export EVE_LOCAL_ZARR_DIR=${ZARR_OUT}
export EVE_HYDRA_RUN_DIR=${HYDRA_RUN_DIR}
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12350
export NCCL_NET="AWS Libfabric"

echo "[step 2] Launching trainHydra debug"
cd ${HYDRA_RUN_DIR}
python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_cup_bc \
    name=eve_cup_bc_debug_'"${SLURM_JOB_ID}"' \
    description=eve_cup_bc_debug_'"${SLURM_JOB_ID}"' \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    trainer.num_nodes=1 \
    launch_params.gpus_per_node=1 \
    launch_params.nodes=1

echo "[done] Listing validation videos:"
find ${HYDRA_RUN_DIR}/videos -name "validation_video*.mp4" 2>/dev/null | head -5
echo "[done] '"${SLURM_JOB_ID}"'"
'
