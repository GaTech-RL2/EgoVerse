#!/bin/bash
#SBATCH --job-name=T_cup_BC2ID8EV_DEBUG
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/debug/slurm-cup-BC2ID8EV-debug-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/debug/slurm-cup-BC2ID8EV-debug-%j.err
#SBATCH --partition=normal
#SBATCH --requeue
#SBATCH --signal=USR1@600
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

set -eo pipefail
ulimit -c 0

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main

export EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/cup_full_zarr}
export ARIA_LOCAL_ZARR_DIR=${ARIA_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/aria_cup_2id_8ev}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/DEBUG/eve_aria_cup_cotrain_BC+2ID+8EV/${SLURM_JOB_ID}}

mkdir -p "${HYDRA_RUN_DIR}" /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/debug

echo "[cup BC+2ID+8EV DEBUG] full mix indomain + 4 EgoVerse labs"

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
export MASTER_PORT=12355
export NCCL_NET="AWS Libfabric"

cd ${HYDRA_RUN_DIR}
python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_aria_cup_cotrain \
    name=eve_aria_cup_cotrain_BC2ID8EV_debug_'"${SLURM_JOB_ID}"' \
    description=eve_aria_cup_cotrain_BC2ID8EV_debug_'"${SLURM_JOB_ID}"' \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    trainer.num_nodes=1 \
    launch_params.gpus_per_node=1 \
    launch_params.nodes=1

find ${HYDRA_RUN_DIR}/videos -name "validation_video*.mp4" 2>/dev/null | head -5
'
