#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
# 250G: 2 ranks; the freeform fold set is only 462 episodes (vs 41.6k for
# all-mecka, which needed 500G at 4 ranks) so the CoW tree is small.
#SBATCH --mem=250G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=24:00:00
#SBATCH --job-name=camframe_fold_freeform
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/camframe_fold_clothes_freeform_fit_test_flat5e-5_%j.log

# pi0.5 fit test on the aidan/eval_gating recipe (6c3913fc): CAM (head)-frame cartesian actions with the continuous 6D
# rotation (cartesian_6d, left-wrist fix, padded 20-dim proprio)
# on the mecka FREEFORM fold-clothes episodes (SQL task=folding_clothes,
# 462 eps), from the pi05 base weights. Flat lr 5e-5 (scheduler=null),
# 64/GPU x 2xH200 = global 128. Cadence (epoch = limit_train_batches = 100
# steps): val every 50 epochs, viz every 100 (both heads, <=10 batches),
# ckpt every 50 epochs (top-3 by step + last).
# Eval = eval_pi_camframe_6d (actions already in cam frame: the revert only
# converts 6D -> ypr, so the cam-frame MSE is comparable to the wrist-frame
# twin) + the train_viz second head (train_viz_pi_camframe_6d, new here).
#
# Requires the CPU norm job's output (norm_fold_freeform_cam6d.sbatch):
#   /storage/project/r-dxu345-0/agao81/norm_stats/fold_freeform_cam6d/norm_stats
# Submit with --dependency=afterok:<norm job id>.
#
# Runs from the aidan/fold-freeform-frames worktree (code = eval_gating +
# these configs) so the primary checkout stays untouched. PYTHONPATH pins
# `import egomimic` to the worktree (the emimic venv's editable install
# points at the primary checkout); hydra outputs are forced into the primary
# checkout's logs/ so checkpoints outlive the worktree.
#
# Env knobs:
#   NORM_PATH  - precomputed norm_stats dir (default above)
#   CKPT_PATH  - resume from a Lightning checkpoint
#   LR         - optimizer lr (default 5e-5)

set -euo pipefail
WT=/storage/project/r-dxu345-0/agao81/EgoVerse/.claude/worktrees/fold-freeform-frames
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

NORM_PATH="${NORM_PATH-/storage/project/r-dxu345-0/agao81/norm_stats/fold_freeform_cam6d/norm_stats}"
CKPT_PATH="${CKPT_PATH-}"
LR="${LR-5e-5}"
LABEL=camframe_fold_clothes_freeform_fit_test_flat5e-5

if [[ ! -f "$NORM_PATH/norm_stats.json" ]]; then
  echo "ERROR: precomputed norm stats not found: $NORM_PATH/norm_stats.json" >&2
  echo "Run egomimic/scripts/norm_fold_freeform_cam6d.sbatch first." >&2
  exit 1
fi

ARGS=(
  --config-name=train_zarr_cartesian_pi
  data=mecka_fold_freeform_pi_cam6d
  model=pi0.5_bc_mecka_6d
  evaluator=eval_pi_camframe_6d
  # eval_pi_camframe_6d lacks the render cap eval_pi_wristframe_6d carries;
  # match it so both frames render <=10 val batches per viz epoch.
  +evaluator.viz_max_batches=10
  # Second eval head: metrics + videos on TRAIN data each val (train_viz/).
  +evaluator@train_viz_evaluator=train_viz_pi_camframe_6d
  launch_params.gpus_per_node=2
  "model.optimizer.lr=$LR"
  # Flat LR: drop pi0.5_base's constant-with-warmup (2000 warmup steps).
  model.scheduler=null
  "norm_stats.precomputed_norm_path=$NORM_PATH"
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch
  trainer.check_val_every_n_epoch=50
  callbacks.model_checkpoint.every_n_epochs=50
  "name=$LABEL"
  description=bs64x2_flat5e-5_cam6d
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
