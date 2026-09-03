#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=72:00:00
# atl1-1-02-012-16-0 GPU-1776d65d and atl1-1-02-014-9-0 GPU-8375b1cd were
# thermally throttled (93 C, SM ~350-420 MHz) on 2026-09-01 and clamped the
# first-round runs to ~3 eps/h; keep the sweep off those nodes.
#SBATCH --exclude=atl1-1-02-012-16-0,atl1-1-02-014-9-0
#SBATCH --job-name=wristframe_fold_freeform_opsplit
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/wristframe_fold_clothes_freeform_fit_test_flat5e-5_opsplit_%j.log

# pi0.5 fit test on the aidan/eval_gating recipe, held-out-OPERATOR split:
# WRIST-frame cartesian actions, continuous 6D rotation (cartesian_wristframe_6d,
# left-wrist fix, padded 20-dim proprio)
# on the mecka FREEFORM fold-clothes episodes (423 train eps / 39 val eps from
# 25 held-out operators), from the pi05 base weights. Flat lr 5e-5
# (scheduler=null), 64/GPU x 2xH200 = global 128. Cadence (epoch =
# limit_train_batches = 100 steps): val every 20 epochs (Aidan, 2026-09-01), viz
# every 100 (both heads, <=10 batches; 100 is a multiple of 20 so viz still lands
# on a val epoch), ckpt every 50 epochs (top-3 by step + last).
# Eval = eval_pi_wristframe_6d (6D -> ypr, wrist -> cam revert) + train_viz head.
#
# Requires the CPU norm job's output (norm_fold_freeform_opsplit_wrist6d.sbatch):
#   /storage/project/r-dxu345-0/agao81/norm_stats/fold_freeform_opsplit_wrist6d/norm_stats
# Submit with --dependency=afterok:<norm job id>.
#
# Runs from the aidan/fold-freeform-keypoints worktree (code = eval_gating +
# fold-freeform-frames + the keypoint/opsplit commit); PYTHONPATH pins
# `import egomimic` to it; hydra outputs are forced into the primary
# checkout's logs/ so checkpoints outlive the worktree.
#
# Env knobs:
#   NORM_PATH  - precomputed norm_stats dir (default above)
#   CKPT_PATH  - resume from a Lightning checkpoint
#   LR         - optimizer lr (default 5e-5)

set -euo pipefail
WT=/storage/project/r-dxu345-0/agao81/EgoVerse/.claude/worktrees/fold-freeform-kp
REPO=/storage/project/r-dxu345-0/agao81/EgoVerse
cd "$WT"
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
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

NORM_PATH="${NORM_PATH-/storage/project/r-dxu345-0/agao81/norm_stats/fold_freeform_opsplit_wrist6d/norm_stats}"
CKPT_PATH="${CKPT_PATH-}"
LR="${LR-5e-5}"
LABEL=wristframe_fold_clothes_freeform_fit_test_flat5e-5_opsplit

if [[ ! -f "$NORM_PATH/norm_stats.json" ]]; then
  echo "ERROR: precomputed norm stats not found: $NORM_PATH/norm_stats.json" >&2
  echo "Run egomimic/scripts/norm_fold_freeform_opsplit_wrist6d.sbatch first." >&2
  exit 1
fi

ARGS=(
  --config-name=train_zarr_cartesian_pi
  data=mecka_fold_freeform_opsplit_pi_6d
  model=pi0.5_bc_mecka_6d
  evaluator=eval_pi_wristframe_6d
  # Second eval head: metrics + videos on TRAIN data each val (train_viz/).
  +evaluator@train_viz_evaluator=train_viz_pi_wristframe_6d
  launch_params.gpus_per_node=2
  "model.optimizer.lr=$LR"
  # Flat LR: drop pi0.5_base's constant-with-warmup (2000 warmup steps).
  model.scheduler=null
  "norm_stats.precomputed_norm_path=$NORM_PATH"
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch
  trainer.check_val_every_n_epoch=20
  callbacks.model_checkpoint.every_n_epochs=50
  "name=$LABEL"
  description=bs64x2_flat5e-5_wrist6d_opsplit
  "+logger.wandb.name=$LABEL"
  # Keep outputs (checkpoints, norm-stats cache) out of the worktree.
  'hydra.run.dir=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/${name}/${description}_${now:%Y-%m-%d_%H-%M-%S}'
)
# Lightning names checkpoints "epoch_epoch=..." — quote so hydra's grammar
# doesn't choke on '='.
[[ -n "$CKPT_PATH" ]] && ARGS+=("ckpt_path='$CKPT_PATH'")

echo "[train] cwd=$(pwd) commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[train] srun python -u egomimic/trainHydra.py ${ARGS[*]}"
srun "$PY" -u egomimic/trainHydra.py "${ARGS[@]}"
