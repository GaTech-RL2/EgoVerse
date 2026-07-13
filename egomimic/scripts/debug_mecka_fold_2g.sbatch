#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=24:00:00
#SBATCH --job-name=pi05_fold_debug
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/pi05_fold_debug_%j.log
#SBATCH --exclude=atl1-1-01-009-16-0

# Debug run: pi0.5 BC on the mecka FLAGSHIP fold-clothes subset (1,607
# episodes, task='fold_clothes'), 2xH200, exercising everything on this
# branch: wristframe-6D + left-wrist fix, wrap-aware cam MSE (+_nowrap),
# xyz/ypr splits, Frechet, reverse-KL M=4 family, DTW, train_viz second
# head, world-size-corrected video fps, val_at_start baseline.
#
# Cadence (epoch = 100 steps): val every 10 epochs, viz every 100 for BOTH
# heads (train_viz inherits the base evaluator's viz gate).

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if grep -q WANDB_API_KEY apikey.txt 2>/dev/null; then
  export WANDB_API_KEY="$(grep -oP 'WANDB_API_KEY=\K.*' apikey.txt)"
  export WANDB_MODE=online
elif grep -q api.wandb.ai ~/.netrc 2>/dev/null; then
  export WANDB_MODE=online
else
  export WANDB_MODE=offline
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NORM_PATH="${NORM_PATH-/storage/project/r-dxu345-0/agao81/norm_stats/mecka_all_6d/norm_stats}"

ARGS=(
  --config-name=train_zarr_cartesian_pi
  data=mecka_fold_flagship_pi_6d
  model=pi0.5_bc_mecka_6d
  evaluator=eval_pi_wristframe_6d
  +evaluator@train_viz_evaluator=train_viz_pi_wristframe_6d
  launch_params.gpus_per_node=2
  trainer.check_val_every_n_epoch=10
  # norm stats from the full mecka set (superset, same distribution) — avoids
  # a recompute for a debug run.
  "norm_stats.precomputed_norm_path=$NORM_PATH"
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch
  callbacks.model_checkpoint.every_n_epochs=50
  name=pi05_fold_debug
  description=pi05_fold_flagship_debug_bs64x2_wrist6d_allmetrics
)

echo "[debug] srun python -u egomimic/trainHydra.py ${ARGS[*]}"
srun "$PY" -u egomimic/trainHydra.py "${ARGS[@]}"
