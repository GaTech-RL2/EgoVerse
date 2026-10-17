#!/bin/bash
#SBATCH --job-name=eve_mug2_BC_1n4g_full
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2/slurm-mug2-BC-1n-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2/slurm-mug2-BC-1n-%j.err
#SBATCH --partition=normal
#SBATCH --requeue
#SBATCH --signal=USR1@600
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaqchen@ethz.ch
##################### SBATCH RESOURCES #####################
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
############################################################
# real_mug v2 (bimanual) BC on 1 node x 4 GPUs (20 sanity episodes -> a single
# node is plenty; batch 32/GPU -> 128 effective, base LR 3e-4).
# EXTRA_HYDRA_ARGS="trainer.max_epochs=1000" to shorten; resumes from last.ckpt.
# Submit with `sbatch -A <account> ...` to override the account above.

set -eo pipefail
ulimit -c 0

EGOMAIN=${EGOMAIN:-/iopsstor/scratch/cscs/jiaqchen/code/EgoVerse-main}
export EGOMAIN
export VARIANT="${VARIANT:-BC}"
export RUN_NAME="${RUN_NAME:-mug2_BC_1n4g_full}"
export WANDB_GROUP="${WANDB_GROUP:-real_mug_v2}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"
export EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/real_mug_v2/zarr}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/FULL_1n/eve_mug2_bc/${RUN_NAME}}
mkdir -p "${HYDRA_RUN_DIR}" /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2
echo "[mug2 BC FULL 1n4g] RUN_NAME=${RUN_NAME}  ZARR=${EVE_LOCAL_ZARR_DIR}  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}  RESTART=${SLURM_RESTART_COUNT:-0}"

LAST_CKPT="${HYDRA_RUN_DIR}/checkpoints/last.ckpt"
if [ -f "${LAST_CKPT}" ]; then CKPT_OVERRIDE="ckpt_path=\"${LAST_CKPT}\""; echo "  Resuming from ${LAST_CKPT}"; else CKPT_OVERRIDE="ckpt_path=null"; echo "  Fresh start"; fi

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml \
     --container-workdir=${EGOMAIN} \
     bash -lc '
set -eo pipefail
EGOMAIN='"${EGOMAIN}"'; export EGOMAIN
HYDRA_RUN_DIR='"${HYDRA_RUN_DIR}"'
source ${EGOMAIN}/eth_clariden/clariden.sh
export EVE_HYDRA_RUN_DIR=${HYDRA_RUN_DIR}
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12372
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export NCCL_NET="AWS Libfabric"
export RLDB_LOAD_WORKERS=16
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
    '"${EXTRA_HYDRA_ARGS:-}"' \
    '"${CKPT_OVERRIDE}"'
'
