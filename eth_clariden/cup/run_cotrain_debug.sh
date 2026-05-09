#!/bin/bash
#SBATCH --job-name=T_eve_aria_cotrain_DEBUG
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/slurm-cotrain-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/slurm-cotrain-%j.err
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

export EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/SMOKE_TEST_PG4/cup_eve_zarr}
export ARIA_LOCAL_ZARR_DIR=${ARIA_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/aria_cup_zarr}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/DEBUG/eve_aria_cotrain/${SLURM_JOB_ID}}

mkdir -p "${HYDRA_RUN_DIR}"

echo "============================================================"
echo "[run_cotrain_debug] EgoVerse-main cotrain debug pipeline"
echo "  EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR}"
echo "  ARIA_LOCAL_ZARR_DIR=${ARIA_LOCAL_ZARR_DIR}"
echo "  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "============================================================"

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml \
     --container-workdir=${EGOMAIN} \
     bash -lc '
set -eo pipefail
EGOMAIN='"${EGOMAIN}"'
HYDRA_RUN_DIR='"${HYDRA_RUN_DIR}"'

source ${EGOMAIN}/eth_clariden/clariden.sh
nvidia-smi --query-gpu=name,memory.total --format=csv || true

export EVE_HYDRA_RUN_DIR=${HYDRA_RUN_DIR}
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12348
export NCCL_NET="AWS Libfabric"

echo "[cotrain] Launching trainHydra"
cd ${HYDRA_RUN_DIR}
python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_aria_cotrain \
    name=eve_aria_cotrain_'"${SLURM_JOB_ID}"' \
    description=eve_aria_cotrain_'"${SLURM_JOB_ID}"' \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    trainer.num_nodes=1 \
    launch_params.gpus_per_node=1 \
    launch_params.nodes=1

echo "[done] '"${SLURM_JOB_ID}"'"
'
