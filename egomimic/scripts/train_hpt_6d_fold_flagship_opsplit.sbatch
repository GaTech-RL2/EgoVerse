#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=hpt_6d_fold_flagship_opsplit
#SBATCH --output=logs/hpt_6d_fold_flagship_opsplit_%j.log

# ~300M HPT flow BC (+ frozen Qwen3-Embedding text stem) on the 18-D
# WRIST-frame 6D-rotation cartesian action (cartesian_wristframe_6d, left-wrist
# fix, padded 20-dim proprio), mecka FLAGSHIP fold-clothes, seen/unseen
# OPERATOR split (data=mecka_fold_flagship_opsplit_hpt_6d):
#   train              18 operators, 95 % of their episodes (~1346 eps)
#   Valid/*            2 UNSEEN operators, all episodes (191 eps)
#   train_viz/Valid/*  the same 18 operators, the held-out 5 % (~70 eps)
# Constant lr 5e-5 (model config), 64/GPU x 1 GPU. Cadence (epoch =
# limit_train_batches = 100 steps): val every 20 epochs, viz every 100 (both
# heads, <=10 batches), ckpt every 50 epochs. Eval = eval_hpt_wristframe_6d
# (6D -> ypr, wrist -> cam revert; cam-frame MSE + overlay video) + the
# train_viz head on the seen-operator val.
#
# GPU: the repo rule puts non-pi runs on L40S (`-p gpu-l40s --gres=gpu:1`;
# `--gres=gpu:l40s:1` is rejected). For an H200 instead:
#   sbatch -p gpu-h200 --gres=gpu:h200:1 egomimic/scripts/train_hpt_6d_fold_flagship_opsplit.sbatch
#
# Requires the CPU norm job's output (norm_fold_flagship_opsplit_hpt_6d.sbatch);
# submit with --dependency=afterok:<norm job id>. Submit from the repo root
# (the #SBATCH --output path is relative to it; `mkdir -p logs` first).
#
# The Qwen stem loads Qwen/Qwen3-Embedding-0.6B from the HF cache; compute
# nodes have no internet, so pre-fetch it once on the login node:
#   emimic/bin/python -c "from transformers import AutoModel, AutoTokenizer; \
#     AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B'); AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')"
#
# Env knobs:
#   REPO       - checkout to run from (default: the submit directory)
#   NORM_ROOT  - where norm-stats dirs live (default: $REPO/norm_stats)
#   DATASET_DIR - zarr mirror (default: the shared PACE mirror /storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets)
#   NORM_PATH  - precomputed norm_stats dir (default: the norm job's output)
#   CKPT_PATH  - resume from a Lightning checkpoint
#   LR         - optimizer lr (default 5e-5)
#   BS         - per-GPU batch size (default 64)

set -euo pipefail
REPO="${REPO-$SLURM_SUBMIT_DIR}"
NORM_ROOT="${NORM_ROOT-$REPO/norm_stats}"
DATASET_DIR="${DATASET_DIR-/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY="$REPO/emimic/bin/python"
export PATH="$REPO/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# wandb online (key from apikey.txt); falls back to offline if absent.
if grep -q WANDB_API_KEY "$REPO/apikey.txt" 2>/dev/null; then
  export WANDB_API_KEY="$(grep -oP 'WANDB_API_KEY=\K.*' "$REPO/apikey.txt")"
  export WANDB_MODE=online
elif grep -q api.wandb.ai ~/.netrc 2>/dev/null; then
  export WANDB_MODE=online   # netrc-authenticated
else
  export WANDB_MODE=offline
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Private compile caches per job (same-user jobs on one node otherwise share
# /tmp/torchinductor_<user>).
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="/tmp/triton_${USER}_${SLURM_JOB_ID}"

NORM_PATH="${NORM_PATH-$NORM_ROOT/fold_flagship_opsplit_hpt_6d/norm_stats}"
CKPT_PATH="${CKPT_PATH-}"
LR="${LR-5e-5}"
BS="${BS-64}"
LABEL=hpt300m_6d_fold_clothes_flagship_opsplit

if [[ ! -f "$NORM_PATH/norm_stats.json" ]]; then
  echo "ERROR: precomputed norm stats not found: $NORM_PATH/norm_stats.json" >&2
  echo "Run egomimic/scripts/norm_fold_flagship_opsplit_hpt_6d.sbatch first." >&2
  exit 1
fi

ARGS=(
  --config-name=train_zarr_mecka_flagship_6d_hpt
  "paths.dataset_dir=${DATASET_DIR}"
  launch_params.gpus_per_node=1
  "model.optimizer.lr=$LR"
  "data.train_dataloader_params.human_bimanual.batch_size=$BS"
  "data.valid_dataloader_params.human_bimanual.batch_size=$BS"
  "data.train_viz_dataloader_params.human_bimanual.batch_size=$BS"
  "norm_stats.precomputed_norm_path=$NORM_PATH"
  trainer.check_val_every_n_epoch=20
  callbacks.model_checkpoint.every_n_epochs=50
  "name=$LABEL"
  "description=bs${BS}x1_const${LR}_6d_opsplit"
  "+logger.wandb.name=$LABEL"
)
# Lightning names checkpoints "epoch_epoch=..." — quote so hydra's grammar
# doesn't choke on '='.
[[ -n "$CKPT_PATH" ]] && ARGS+=("ckpt_path='$CKPT_PATH'")

echo "[train] cwd=$(pwd) commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[train] srun python -u egomimic/trainHydra.py ${ARGS[*]}"
srun "$PY" -u egomimic/trainHydra.py "${ARGS[@]}"
