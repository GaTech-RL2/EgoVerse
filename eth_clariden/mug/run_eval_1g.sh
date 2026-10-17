#!/bin/bash
#SBATCH --job-name=eve_mug_eval_1g
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug/slurm-mug-eval-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug/slurm-mug-eval-%j.err
#SBATCH --partition=normal
##################### SBATCH RESOURCES #####################
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
############################################################
# Single-process validation pass (mode=eval) from a checkpoint: writes the
# validation videos (projected EE chunk + ORCA wireframe + joint panel) without
# the multi-rank write race, plus val metrics. Usage:
#   CKPT=.../checkpoints/epoch_epoch=1499.ckpt sbatch -A cvg-prof-m-2 -p debug run_eval_1g.sh
# Submit with `sbatch -A <account> ...` to override the account above.

set -eo pipefail
ulimit -c 0

EGOMAIN=${EGOMAIN:-/iopsstor/scratch/cscs/jiaqchen/code/EgoVerse-main}
export EGOMAIN

TRAIN_RUN=${TRAIN_RUN:-mug_BC_4n4g_e1000}
TRAIN_DIR=${TRAIN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/FULL_4n/eve_mug_bc/${TRAIN_RUN}}
CKPT=${CKPT:-${TRAIN_DIR}/checkpoints/last.ckpt}
CKPT_TAG=$(basename "${CKPT}" .ckpt | tr -d '=')
export RUN_NAME="${RUN_NAME:-${TRAIN_RUN}_eval_${CKPT_TAG}}"
export EVE_LOCAL_ZARR_DIR=${EVE_LOCAL_ZARR_DIR:-/iopsstor/scratch/cscs/jiaqchen/data/real_mug_30hz/zarr}
export HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/EVAL/eve_mug_bc/${RUN_NAME}}
# reuse the training run's normalization stats so the checkpoint sees identical inputs
NORM_STATS=${NORM_STATS:-${TRAIN_DIR}/norm_stats}
mkdir -p "${HYDRA_RUN_DIR}" /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug
[ -f "${CKPT}" ] || { echo "checkpoint not found: ${CKPT}"; exit 1; }

echo "============================================================"
echo "[mug EVAL 1g] mode=eval from ${CKPT}"
echo "  RUN_NAME=${RUN_NAME}  HYDRA_RUN_DIR=${HYDRA_RUN_DIR}"
echo "  NORM_STATS=${NORM_STATS}"
echo "============================================================"

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml \
     --container-workdir=${EGOMAIN} \
     bash -lc '
set -eo pipefail
EGOMAIN='"${EGOMAIN}"'; export EGOMAIN
HYDRA_RUN_DIR='"${HYDRA_RUN_DIR}"'
NORM_STATS='"${NORM_STATS}"'
CKPT='"${CKPT}"'

source ${EGOMAIN}/eth_clariden/clariden.sh
export EVE_HYDRA_RUN_DIR=${HYDRA_RUN_DIR}
export RLDB_LOAD_WORKERS=8
export HF_HUB_DISABLE_PROGRESS_BARS=1
export WANDB_MODE=offline
# (no ls-globs here: under `set -e -o pipefail` a missing glob makes ls exit 2 and kills the script)
PRECOMPUTED=""
[ -f "${NORM_STATS}/norm_stats.json" ] && PRECOMPUTED="${NORM_STATS}"

cd ${HYDRA_RUN_DIR}
exec python ${EGOMAIN}/egomimic/trainHydra.py \
    --config-name=train_zarr_eve_mug_bc_full \
    mode=eval \
    name='"${RUN_NAME}"' \
    description='"${RUN_NAME}"' \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    launch_params.gpus_per_node=1 \
    launch_params.nodes=1 \
    trainer.num_nodes=1 \
    "ckpt_path=\"${CKPT}\"" \
    ${PRECOMPUTED:+"norm_stats.precomputed_norm_path=\"${PRECOMPUTED}\""} \
    '"${EXTRA_HYDRA_ARGS:-}"'
'
