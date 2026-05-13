#!/bin/bash
#SBATCH --job-name=eve_cup_BC1ID_4n4g_full
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/full_4n/slurm-cup-BC1ID-4n-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/full_4n/slurm-cup-BC1ID-4n-%j.err
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
# Cup BC+1ID: BC + half of indomain Aria (eth lab). Same aria zarr dir as BC+2ID,
# but with valid_ratio=0.6 override so only ~40% of indomain episodes train (= "half").

set -eo pipefail
ulimit -c 0

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main

export VARIANT="${VARIANT:-BC+1ID}"
export RUN_NAME="${RUN_NAME:-cup_BC_1ID_4n4g_full}"
export WANDB_GROUP="${WANDB_GROUP:-cup_on_saucer}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"

export EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/cup_full_zarr}
export ARIA_LOCAL_ZARR_DIR=${ARIA_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/aria_cup_indomain_eth}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/FULL_4n/eve_aria_cup_cotrain/${RUN_NAME}}
mkdir -p "${HYDRA_RUN_DIR}" /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/full_4n

echo "============================================================"
echo "[cup BC+1ID FULL 4n4g] EgoVerse-main 2000-epoch BC+1ID (half indomain), 16 GPUs"
echo "  RUN_NAME=${RUN_NAME}"
echo "  EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR}"
echo "  ARIA_LOCAL_ZARR_DIR=${ARIA_LOCAL_ZARR_DIR}"
echo "  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "  SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"
echo "  Aria valid_ratio override: 0.6 (only ~40% trains -> 1ID)"
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
export MASTER_PORT=12364
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
    data.train_datasets.aria_bimanual.valid_ratio=0.55 \
    data.valid_datasets.aria_bimanual.valid_ratio=0.55 \
    '"${CKPT_OVERRIDE}"'
'
