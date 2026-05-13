#!/bin/bash
#SBATCH --job-name=eve_obj_BC_4n4g_debug
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/obj/slurm-obj-BC-debug-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/obj/slurm-obj-BC-debug-%j.err
#SBATCH --partition=normal
#SBATCH --requeue
#SBATCH --signal=USR1@180
##################### SBATCH RESOURCES #####################
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
############################################################
# 4n4g (16-rank) obj BC debug job. Short walltime, fast epochs, validates that
# the right_arm pipeline boots, trains a couple of epochs, runs a validation
# pass, writes a validation video without crashing, and exits cleanly.

set -eo pipefail
ulimit -c 0

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main

export VARIANT="${VARIANT:-BC}"
export RUN_NAME="${RUN_NAME:-obj_BC_4n4g_debug}"
export WANDB_GROUP="${WANDB_GROUP:-object_in_container_debug}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"

export EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/obj_full_zarr}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/DEBUG_4n/eve_obj_bc/${RUN_NAME}}
mkdir -p "${HYDRA_RUN_DIR}" "$(dirname /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/obj/.)"

echo "============================================================"
echo "[obj BC DEBUG 4n4g] right_arm, validation_video smoke test"
echo "  RUN_NAME=${RUN_NAME}"
echo "  EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR}"
echo "  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "  SLURM_NNODES=${SLURM_NNODES}"
echo "============================================================"

CKPT_OVERRIDE="ckpt_path=null"

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
export MASTER_PORT=12357
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export NCCL_NET="AWS Libfabric"
export RLDB_LOAD_WORKERS=8
export HF_HUB_DISABLE_PROGRESS_BARS=1
export WANDB_MODE=offline

echo "[debug] node=$(hostname) MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$WORLD_SIZE SLURM_PROCID=$SLURM_PROCID"

cd ${HYDRA_RUN_DIR}
exec python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_obj_bc_full \
    name='"${RUN_NAME}"' \
    description='"${RUN_NAME}"' \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    trainer.num_nodes=${SLURM_NNODES} \
    launch_params.gpus_per_node=${SLURM_GPUS_PER_NODE:-4} \
    launch_params.nodes=${SLURM_NNODES} \
    trainer.max_epochs=4 \
    trainer.min_epochs=1 \
    trainer.limit_train_batches=10 \
    trainer.limit_val_batches=4 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.model_checkpoint.every_n_epochs=2 \
    '"${CKPT_OVERRIDE}"'
'
