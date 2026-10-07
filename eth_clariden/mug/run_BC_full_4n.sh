#!/bin/bash
#SBATCH --job-name=eve_mug_BC_4n4g_full
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug/slurm-mug-BC-4n-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug/slurm-mug-BC-4n-%j.err
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
# Production 4n x 4g real_mug BC (16 GPUs), same recipe as obj/run_BC_full_4n.sh:
# eve_right_arm, 6D cartesian + 17D hand, batch 32/GPU -> 512 effective, LR 6e-4.
# Submit with `sbatch -A <account> ...` to override the account above.
# Checkpoints are mirrored to capstor store at the end (iopsstor scratch is purged).

set -eo pipefail
ulimit -c 0

EGOMAIN=${EGOMAIN:-/iopsstor/scratch/cscs/jiaqchen/code/EgoVerse-main}
export EGOMAIN

export VARIANT="${VARIANT:-BC}"
export RUN_NAME="${RUN_NAME:-mug_BC_4n4g_full}"
export WANDB_GROUP="${WANDB_GROUP:-real_mug}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"

export EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/real_mug_30hz/zarr}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/FULL_4n/eve_mug_bc/${RUN_NAME}}
CAPSTOR_MIRROR=${CAPSTOR_MIRROR:-/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/egomim_out/mug/${RUN_NAME}}
mkdir -p "${HYDRA_RUN_DIR}" /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug

echo "============================================================"
echo "[mug BC FULL 4n4g] 2000-epoch right_arm BC, 16 GPUs"
echo "  RUN_NAME=${RUN_NAME}"
echo "  EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR}"
echo "  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "  CAPSTOR_MIRROR=${CAPSTOR_MIRROR}"
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

# Mirror checkpoints + configs to capstor store on exit (success, failure or requeue signal).
mirror() {
    echo "[mirror] rsync ${HYDRA_RUN_DIR} -> ${CAPSTOR_MIRROR}"
    mkdir -p "${CAPSTOR_MIRROR}" && rsync -a --exclude 'wandb' --exclude 'videos' "${HYDRA_RUN_DIR}/" "${CAPSTOR_MIRROR}/" || echo "[mirror] rsync failed (quota?)"
}
trap mirror EXIT

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml \
     --container-workdir=${EGOMAIN} \
     bash -lc '
set -eo pipefail
EGOMAIN='"${EGOMAIN}"'; export EGOMAIN
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

cd ${HYDRA_RUN_DIR}
exec python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_mug_bc_full \
    name='"${RUN_NAME}"' \
    description='"${RUN_NAME}"' \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    trainer.num_nodes=${SLURM_NNODES} \
    launch_params.gpus_per_node=${SLURM_GPUS_PER_NODE:-4} \
    launch_params.nodes=${SLURM_NNODES} \
    model.optimizer.lr=6e-4 \
    '"${CKPT_OVERRIDE}"'
'
