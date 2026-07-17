#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=72:00:00
#SBATCH --job-name=pi05_aria_grip_eh
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/pi05_aria_grip_eh_%j.log
# atl1-1-01-009-16-0: 5x backward slowdown at 8 ranks; excluded on principle.
#SBATCH --exclude=atl1-1-01-009-16-0

# pi0.5 finetune on aria fold_clothes (709 episodes) WITH the keypoint
# pseudo-gripper, 4xH200, initialized from the EVA+HUMAN base (abc-eva ft epoch-399, itself from mecka-849)
# (weights only). 20D actions/proprio (grips at robot slots 9/19), robot
# converter, wristframe-6D, all branch val metrics + train_viz head.
# NO fix_mecka_left_wrist: aria natively has the target convention
# (verified on-data; the fix would double-flip).
#
# Cadence (epoch = 100 steps): val every 50 epochs, viz every 100 (both
# heads), ckpt every 50 epochs top-3 by step.
#
# Env knobs: INIT_CKPT / NORM_PATH / CKPT_PATH (resume) / LR (default 5e-5)

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

INIT_CKPT="${INIT_CKPT-/storage/project/r-dxu345-0/agao81/EgoVerse/logs/pi05_abc_eva_ft/pi05_abc_eva_ft_from_mecka849_bs64x8_wrist6d_2026-07-14_11-47-33/checkpoints/epoch_epoch=399.ckpt}"
NORM_PATH="${NORM_PATH-/storage/project/r-dxu345-0/agao81/norm_stats/aria_fold_grip_6d/norm_stats}"
CKPT_PATH="${CKPT_PATH-}"
LR="${LR-5e-5}"

if [[ ! -f "$NORM_PATH/norm_stats.json" ]]; then
  echo "ERROR: precomputed norm stats not found: $NORM_PATH/norm_stats.json" >&2
  exit 1
fi
if [[ ! -f "$INIT_CKPT" ]]; then
  echo "ERROR: init checkpoint not found: $INIT_CKPT" >&2
  exit 1
fi

ARGS=(
  --config-name=train_zarr_cartesian_pi
  data=aria_fold_pi_6d_grip
  model=pi0.5_ft_aria_fold_grip
  evaluator=eval_pi_wristframe_6d_grip
  +evaluator@train_viz_evaluator=train_viz_pi_wristframe_6d_grip
  launch_params.gpus_per_node=4
  "model.optimizer.lr=$LR"
  "norm_stats.precomputed_norm_path=$NORM_PATH"
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch
  callbacks.model_checkpoint.every_n_epochs=50
  name=pi05_aria_fold_grip_evahuman
  description=pi05_aria_fold_grip_ft_from_evahuman399_bs64x4_wrist6d
)
if [[ -n "$CKPT_PATH" ]]; then
  ARGS+=("ckpt_path='$CKPT_PATH'")
else
  ARGS+=("+init_weights_ckpt='$INIT_CKPT'")
fi

echo "[train] srun python -u egomimic/trainHydra.py ${ARGS[*]}"
srun "$PY" -u egomimic/trainHydra.py "${ARGS[@]}"
