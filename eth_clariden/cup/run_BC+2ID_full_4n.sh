#!/bin/bash
#SBATCH --job-name=eve_cup_BC2ID_4n4g_full
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/full_4n/slurm-cup-BC2ID-4n-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/full_4n/slurm-cup-BC2ID-4n-%j.err
#SBATCH --partition=normal
#SBATCH --requeue
#SBATCH --signal=USR1@600
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaqchen@ethz.ch
##################### SBATCH RESOURCES #####################
#SBATCH --time=18:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
############################################################
# Cup BC+2ID: BC + FULL indomain Aria (eth lab). Default valid_ratio.

set -eo pipefail
ulimit -c 0

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main

export VARIANT="${VARIANT:-BC+2ID}"
export RUN_NAME="${RUN_NAME:-cup_BC_2ID_4n4g_full}"
export WANDB_GROUP="${WANDB_GROUP:-cup_on_saucer}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"

export EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/cup_full_zarr}
export ARIA_LOCAL_ZARR_DIR=${ARIA_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/aria_cup_indomain_eth}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/FULL_4n/eve_aria_cup_cotrain/${RUN_NAME}}
mkdir -p "${HYDRA_RUN_DIR}" /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/full_4n

echo "============================================================"
echo "[cup BC+2ID FULL 4n4g] EgoVerse-main 2000-epoch BC+2ID (full indomain), 16 GPUs"
echo "  RUN_NAME=${RUN_NAME}"
echo "  ARIA_LOCAL_ZARR_DIR=${ARIA_LOCAL_ZARR_DIR}"
echo "  SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"
echo "============================================================"

LAST_CKPT="${HYDRA_RUN_DIR}/checkpoints/last.ckpt"
if [ -f "${LAST_CKPT}" ]; then
    CKPT_OVERRIDE="ckpt_path=\"${LAST_CKPT}\""
else
    CKPT_OVERRIDE="ckpt_path=null"
fi

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml \
     --container-workdir=${EGOMAIN} \
     bash -lc '
set -eo pipefail
EGOMAIN='"${EGOMAIN}"'
HYDRA_RUN_DIR='"${HYDRA_RUN_DIR}"'

source ${EGOMAIN}/eth_clariden/clariden.sh
nvidia-smi --query-gpu=name,memory.total --format=csv || true

export EVE_HYDRA_RUN_DIR=${HYDRA_RUN_DIR}
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12365
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export NCCL_NET="AWS Libfabric"
export RLDB_LOAD_WORKERS=32
export HF_HUB_DISABLE_PROGRESS_BARS=1
export WANDB_MODE=offline

cd ${HYDRA_RUN_DIR}
exec python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_aria_cup_cotrain_full \
    name='"${RUN_NAME}"' \
    description='"${RUN_NAME}"' \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    trainer.num_nodes=${SLURM_NNODES} \
    launch_params.gpus_per_node=${SLURM_GPUS_PER_NODE:-4} \
    launch_params.nodes=${SLURM_NNODES} \
    model.optimizer.lr=6e-4 \
    '"${CKPT_OVERRIDE}"'
'
