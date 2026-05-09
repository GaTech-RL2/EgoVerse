#!/bin/bash
#SBATCH --job-name=T_eve_main_BC_DEBUG
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/slurm-cup-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/slurm-cup-%j.err
#SBATCH --partition=normal
#SBATCH --requeue
#SBATCH --signal=USR1@600
##################### SBATCH RESOURCES #####################
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
############################################################

set -eo pipefail
ulimit -c 0

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main

# Inputs / outputs (set before srun so they propagate via --export=ALL)
export LEROBOT_ROOT=${LEROBOT_ROOT:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/SMOKE_TEST_PG4/cup_lerobot_base_frame}
export ZARR_OUT=${ZARR_OUT:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/SMOKE_TEST_PG4/cup_eve_zarr}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/DEBUG/eve_bc/${SLURM_JOB_ID}}

mkdir -p "${HYDRA_RUN_DIR}" "$(dirname "${ZARR_OUT}")"

echo "============================================================"
echo "[run_BC_debug_main] EgoVerse-main BC debug pipeline"
echo "  LEROBOT_ROOT=${LEROBOT_ROOT}"
echo "  ZARR_OUT=${ZARR_OUT}"
echo "  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "============================================================"

# Single srun runs both conversion and training inside the container
# (CSCS recommends --environment on srun, not on sbatch).
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
python --version || true

##################### STEP 1: lerobot -> zarr #####################
if [ -d "${ZARR_OUT}" ] && compgen -G "${ZARR_OUT}/*.zarr" > /dev/null; then
    echo "[step 1] Found existing zarr episodes in ${ZARR_OUT}; skipping conversion"
else
    echo "[step 1] Converting LeRobot -> Zarr"
    python ${EGOMAIN}/egomimic/scripts/eve_process/lerobot_to_zarr.py \
        --root ${LEROBOT_ROOT} \
        --output-dir ${ZARR_OUT} \
        --embodiment eve_bimanual \
        --task-name put_cup_on_saucer \
        --task-description "EVE bimanual put cup on saucer (debug)"
fi

ls -la ${ZARR_OUT}

##################### STEP 2: train #####################
export EVE_LOCAL_ZARR_DIR=${ZARR_OUT}
export EVE_HYDRA_RUN_DIR=${HYDRA_RUN_DIR}
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12347
export NCCL_NET="AWS Libfabric"

echo "[step 2] Launching trainHydra"
cd ${HYDRA_RUN_DIR}
python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_bc \
    name=eve_bc_debug_'"${SLURM_JOB_ID}"' \
    description=eve_bc_debug_'"${SLURM_JOB_ID}"' \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    trainer.num_nodes=1 \
    launch_params.gpus_per_node=1 \
    launch_params.nodes=1

echo "[done] '"${SLURM_JOB_ID}"'"
'
