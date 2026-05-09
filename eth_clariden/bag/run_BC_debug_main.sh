#!/bin/bash
#SBATCH --job-name=T_eve_bag_BC_DEBUG
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/bag/slurm-bag-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/bag/slurm-bag-%j.err
#SBATCH --partition=normal
#SBATCH --requeue
#SBATCH --signal=USR1@600
##################### SBATCH RESOURCES #####################
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
############################################################

set -eo pipefail
ulimit -c 0

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main

# Inputs
export LEROBOT_ROOT=${LEROBOT_ROOT:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/bag_fixed_lerobot/bag_lerobot_base_frame_pg4_fixed}
export ZARR_OUT=${ZARR_OUT:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/bag_fixed_zarr}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/DEBUG/eve_bag_bc/${SLURM_JOB_ID}}

mkdir -p "${HYDRA_RUN_DIR}" "$(dirname "${ZARR_OUT}")"

echo "============================================================"
echo "[run_BC_debug_main bag] EgoVerse-main bag BC debug pipeline"
echo "  LEROBOT_ROOT=${LEROBOT_ROOT}"
echo "  ZARR_OUT=${ZARR_OUT}"
echo "  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID}"
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

##################### STEP 1: lerobot -> zarr #####################
if [ -d "${ZARR_OUT}" ] && compgen -G "${ZARR_OUT}/*.zarr" > /dev/null; then
    echo "[step 1] Found existing zarr episodes in ${ZARR_OUT}; skipping conversion"
else
    echo "[step 1] Converting LeRobot -> Zarr (with wrist cameras)"
    python ${EGOMAIN}/egomimic/scripts/eve_process/lerobot_to_zarr.py \
        --root ${LEROBOT_ROOT} \
        --output-dir ${ZARR_OUT} \
        --embodiment eve_bimanual \
        --task-name bag_groceries \
        --task-description "EVE bimanual bag groceries (debug, PG=4)"
fi
ls -la ${ZARR_OUT}

##################### STEP 2: train #####################
export EVE_LOCAL_ZARR_DIR=${ZARR_OUT}
export EVE_HYDRA_RUN_DIR=${HYDRA_RUN_DIR}
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12349
export NCCL_NET="AWS Libfabric"

echo "[step 2] Launching trainHydra"
cd ${HYDRA_RUN_DIR}
python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_bag_bc \
    name=eve_bag_bc_debug_'"${SLURM_JOB_ID}"' \
    description=eve_bag_bc_debug_'"${SLURM_JOB_ID}"' \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    trainer.num_nodes=1 \
    launch_params.gpus_per_node=1 \
    launch_params.nodes=1

echo "[done] '"${SLURM_JOB_ID}"'"
'
