#!/bin/bash
#SBATCH --job-name=T_eve_main_BC_DEBUG
#SBATCH --account=a144
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/slurm-cup-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/slurm-cup-%j.err
#SBATCH --partition=debug
#SBATCH --environment=/users/jiaqchen/.edf/faive2lerobot.toml
#SBATCH --requeue
#SBATCH --signal=USR1@600
##################### SBATCH RESOURCES #####################
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
############################################################

set -eo pipefail
ulimit -c 0

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main

# Source venv + PYTHONPATH for EgoVerse-main
source ${EGOMAIN}/eth_clariden/clariden.sh

# Inputs
LEROBOT_ROOT=${LEROBOT_ROOT:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/SMOKE_TEST_PG4/cup_lerobot_base_frame}
ZARR_OUT=${ZARR_OUT:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/SMOKE_TEST_PG4/cup_eve_zarr}
HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/DEBUG/eve_bc/${SLURM_JOB_ID}}

mkdir -p "${HYDRA_RUN_DIR}" "$(dirname "${ZARR_OUT}")"

echo "============================================================"
echo "[run_BC_debug_main] EgoVerse-main BC debug pipeline"
echo "  LEROBOT_ROOT=${LEROBOT_ROOT}"
echo "  ZARR_OUT=${ZARR_OUT}"
echo "  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv || true

##################### STEP 1: lerobot -> zarr #####################
# Skip conversion if the output dir already has any .zarr children.
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

# DDP env (single GPU still works with these set)
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12347
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export NCCL_NET="AWS Libfabric"

echo "[step 2] Launching trainHydra"
cd ${HYDRA_RUN_DIR}
python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_bc \
    name=eve_bc_debug_${SLURM_JOB_ID} \
    description=eve_bc_debug_${SLURM_JOB_ID} \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    trainer.num_nodes=${SLURM_NNODES} \
    launch_params.gpus_per_node=1 \
    launch_params.nodes=1

echo "[done] ${SLURM_JOB_ID}"
