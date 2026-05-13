#!/bin/bash
#SBATCH --job-name=eve_bag_BC_4n4g_test
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/bag/multinode_test/slurm-mn-test-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/bag/multinode_test/slurm-mn-test-%j.err
#SBATCH --partition=normal
#SBATCH --requeue
#SBATCH --signal=USR1@120
##################### SBATCH RESOURCES #####################
#SBATCH --time=00:15:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
############################################################
# 4-node x 4-GPU (16 ranks) multinode requeue smoke test.
# Goals:
#   1) Verify multinode DDP boots across 16 ranks with NCCL/Libfabric.
#   2) Verify SIGUSR1 reaches each rank's python process — *not* just bash —
#      under multi-task-per-node srun. We use `exec python` so bash replaces
#      itself with python; SLURM signals the task PID directly which IS python.
#   3) Verify Lightning's SLURMEnvironment saves hpc_ckpt + scontrol requeue.
#   4) Verify the requeued job auto-resumes from hpc_ckpt with the same wandb run id.

set -eo pipefail
ulimit -c 0

EGOMAIN=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main

# Stable identifiers (do NOT include SLURM_JOB_ID — that changes on requeue).
export VARIANT="${VARIANT:-BC}"
export RUN_NAME="${RUN_NAME:-bag_BC_4n4g_mn_test}"
export WANDB_GROUP="${WANDB_GROUP:-bag_groceries_mn_test}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"

export EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/bag_full_zarr}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/multinode_test/eve_bag_bc/${RUN_NAME}}
mkdir -p "${HYDRA_RUN_DIR}" "$(dirname /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/bag/multinode_test/.)"

echo "============================================================"
echo "[bag BC 4n4g multinode test] EgoVerse-main, 16 ranks, exec python"
echo "  RUN_NAME=${RUN_NAME}"
echo "  EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR}"
echo "  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "  WANDB_RUN_ID=${WANDB_RUN_ID}"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "  SLURM_NNODES=${SLURM_NNODES}"
echo "  SLURM_NODELIST=${SLURM_NODELIST}"
echo "  SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"
echo "============================================================"

LAST_CKPT="${HYDRA_RUN_DIR}/checkpoints/last.ckpt"
if [ -f "${LAST_CKPT}" ]; then
    CKPT_OVERRIDE="ckpt_path=\"${LAST_CKPT}\""
    echo "  Resuming from ${LAST_CKPT}"
else
    CKPT_OVERRIDE="ckpt_path=null"
    echo "  Fresh start (no last.ckpt found)"
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
export MASTER_PORT=12362
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export NCCL_NET="AWS Libfabric"
export RLDB_LOAD_WORKERS=16
export HF_HUB_DISABLE_PROGRESS_BARS=1
export WANDB_MODE=offline

echo "[debug] node=$(hostname) MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$WORLD_SIZE SLURM_PROCID=$SLURM_PROCID"

cd ${HYDRA_RUN_DIR}
exec python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_bag_bc_full \
    name='"${RUN_NAME}"' \
    description='"${RUN_NAME}"' \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    trainer.num_nodes=${SLURM_NNODES} \
    launch_params.gpus_per_node=${SLURM_GPUS_PER_NODE:-4} \
    launch_params.nodes=${SLURM_NNODES} \
    trainer.max_epochs=200 \
    trainer.min_epochs=1 \
    trainer.limit_train_batches=20 \
    trainer.limit_val_batches=4 \
    trainer.check_val_every_n_epoch=10 \
    callbacks.model_checkpoint.every_n_epochs=5 \
    '"${CKPT_OVERRIDE}"'
'
