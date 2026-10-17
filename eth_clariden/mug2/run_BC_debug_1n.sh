#!/bin/bash
#SBATCH --job-name=eve_mug2_BC_1n4g_debug
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2/slurm-mug2-BC-debug-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2/slurm-mug2-BC-debug-%j.err
#SBATCH --partition=normal
#SBATCH --signal=USR1@180
##################### SBATCH RESOURCES #####################
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
############################################################
# 1n4g real_mug v2 (bimanual) BC smoke test: boots the eve_bimanual + wrist-cam
# pipeline, trains a few short epochs, validates + renders both-hand wireframes.
# Submit with `sbatch -A <account> ...` to override the account above.

set -eo pipefail
ulimit -c 0

EGOMAIN=${EGOMAIN:-/iopsstor/scratch/cscs/jiaqchen/code/EgoVerse-main}
export EGOMAIN
export VARIANT="${VARIANT:-BC}"
export RUN_NAME="${RUN_NAME:-mug2_BC_1n4g_debug}"
export WANDB_GROUP="${WANDB_GROUP:-real_mug_v2_debug}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"
export EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/real_mug_v2/zarr}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/DEBUG/eve_mug2_bc/${RUN_NAME}}
mkdir -p "${HYDRA_RUN_DIR}" /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2
echo "[mug2 BC DEBUG 1n4g] RUN_NAME=${RUN_NAME}  ZARR=${EVE_LOCAL_ZARR_DIR}  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml \
     --container-workdir=${EGOMAIN} \
     bash -lc '
set -eo pipefail
EGOMAIN='"${EGOMAIN}"'; export EGOMAIN
HYDRA_RUN_DIR='"${HYDRA_RUN_DIR}"'
source ${EGOMAIN}/eth_clariden/clariden.sh
export EVE_HYDRA_RUN_DIR=${HYDRA_RUN_DIR}
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12371
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export NCCL_NET="AWS Libfabric"
export RLDB_LOAD_WORKERS=8
export HF_HUB_DISABLE_PROGRESS_BARS=1
export WANDB_MODE=offline
cd ${HYDRA_RUN_DIR}
exec python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_mug2_bc_full \
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
    '"${EXTRA_HYDRA_ARGS:-}"' \
    ckpt_path=null
'
