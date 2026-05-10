#!/bin/bash
#SBATCH --job-name=eve_aria_bag_BC_2ID_8EV_pg4_full
#SBATCH --account=a144
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/bag/full/slurm-bag-cotrain-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/bag/full/slurm-bag-cotrain-%j.err
#SBATCH --partition=normal
#SBATCH --requeue
#SBATCH --signal=USR1@600
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaqchen@ethz.ch
##################### SBATCH RESOURCES #####################
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
############################################################

set -eo pipefail
ulimit -c 0

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main

export VARIANT="${VARIANT:-BC+2ID+8EV}"
export RUN_NAME="${RUN_NAME:-bag_BC_2ID_8EV_pg4_full_${SLURM_JOB_ID}}"
export WANDB_GROUP="${WANDB_GROUP:-bag_groceries}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"

export EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/bag_full_zarr}
export ARIA_LOCAL_ZARR_DIR=${ARIA_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/aria_bag_zarr}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/FULL/eve_aria_bag_cotrain/${RUN_NAME}}
mkdir -p "${HYDRA_RUN_DIR}" "$(dirname /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/bag/full/.)"

echo "============================================================"
echo "[bag cotrain FULL] EgoVerse-main 2000-epoch BC+2ID+8EV pipeline"
echo "  RUN_NAME=${RUN_NAME}"
echo "  EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR}"
echo "  ARIA_LOCAL_ZARR_DIR=${ARIA_LOCAL_ZARR_DIR}"
echo "  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "  WANDB_RUN_ID=${WANDB_RUN_ID}"
echo "============================================================"

LAST_CKPT="${HYDRA_RUN_DIR}/checkpoints/last.ckpt"
if [ -f "${LAST_CKPT}" ]; then
    CKPT_OVERRIDE="ckpt_path=\"${LAST_CKPT}\""
    echo "  Resuming from ${LAST_CKPT}"
else
    CKPT_OVERRIDE="ckpt_path=null"
    echo "  Fresh start"
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
export MASTER_PORT=12352
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export NCCL_NET="AWS Libfabric"
export RLDB_LOAD_WORKERS=32
export HF_HUB_DISABLE_PROGRESS_BARS=1
export WANDB_MODE=offline

cd ${HYDRA_RUN_DIR}
python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_aria_bag_cotrain_full \
    name='"${RUN_NAME}"' \
    description='"${RUN_NAME}"' \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    trainer.num_nodes=${SLURM_NNODES} \
    launch_params.gpus_per_node=${SLURM_GPUS_PER_NODE:-4} \
    launch_params.nodes=${SLURM_NNODES} \
    '"${CKPT_OVERRIDE}"'
'
