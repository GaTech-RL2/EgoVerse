#!/bin/bash
#SBATCH --job-name=eve_aria_obj_cotrain_4n4g_full
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/obj/slurm-obj-cotrain-4n-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/obj/slurm-obj-cotrain-4n-%j.err
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
# Production 4-node x 4-GPU object_in_container cotrain (BC + 2ID + 8EV).
# right_arm: eve_right_arm + aria_right_arm. 16 GPUs total.
# Per-dataset batch 32 (eve + aria interleaved) => effective ~ 1024 samples/step.
# LR scaled sqrt(4) = 2x: 3e-4 -> 6e-4.

set -eo pipefail
ulimit -c 0

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main

export VARIANT="${VARIANT:-BC+2ID+8EV}"
export RUN_NAME="${RUN_NAME:-obj_BC_2ID_8EV_4n4g_full}"
export WANDB_GROUP="${WANDB_GROUP:-object_in_container}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"

export EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/obj_full_zarr}
export ARIA_LOCAL_ZARR_DIR=${ARIA_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/aria_obj_2id_8ev}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/FULL_4n/eve_aria_obj_cotrain/${RUN_NAME}}
mkdir -p "${HYDRA_RUN_DIR}" "$(dirname /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/obj/.)"

echo "============================================================"
echo "[obj cotrain FULL 4n4g] BC+2ID+8EV, right_arm, 16 GPUs"
echo "  RUN_NAME=${RUN_NAME}"
echo "  EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR}"
echo "  ARIA_LOCAL_ZARR_DIR=${ARIA_LOCAL_ZARR_DIR}"
echo "  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "  WANDB_RUN_ID=${WANDB_RUN_ID}"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "  SLURM_NNODES=${SLURM_NNODES}"
echo "  SLURM_NODELIST=${SLURM_NODELIST}"
echo "  SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"
echo "  LR override: 6e-4 (sqrt(4) scale of 3e-4)"
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
export MASTER_PORT=12356
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export NCCL_NET="AWS Libfabric"
export RLDB_LOAD_WORKERS=32
export HF_HUB_DISABLE_PROGRESS_BARS=1
export WANDB_MODE=offline

echo "[debug] node=$(hostname) MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$WORLD_SIZE SLURM_PROCID=$SLURM_PROCID"

cd ${HYDRA_RUN_DIR}
exec python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_aria_obj_cotrain_full \
    name='"${RUN_NAME}"' \
    description='"${RUN_NAME}"' \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    trainer.num_nodes=${SLURM_NNODES} \
    launch_params.gpus_per_node=${SLURM_GPUS_PER_NODE:-4} \
    launch_params.nodes=${SLURM_NNODES} \
    model.optimizer.lr=6e-4 \
    '"${CKPT_OVERRIDE}"'
'
