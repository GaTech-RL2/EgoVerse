#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G
#SBATCH --gres=gpu:h200:8
#SBATCH --time=72:00:00
#SBATCH --job-name=pi05_abc_eva_ft
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/pi05_abc_eva_ft_%j.log
# atl1-1-01-009-16-0: 5x backward slowdown at 8 ranks (matrix probe 10923198).
#SBATCH --exclude=atl1-1-01-009-16-0

# Stage 3 of the abc-eva finetune chain: pi0.5 finetune on the abc EVA robot
# collection (18,456 eva_bimanual episodes), 8xH200, initialized from the
# mecka-pretrained checkpoint (weights only — fresh optimizer/schedule).
# Wristframe-6D actions + 6D proprio, all branch val metrics + the train_viz
# second head.
#
# Cadence (epoch = 100 steps): val every 50 epochs, viz every 100 (both
# heads), ckpt every 50 epochs top-3 by step.
#
# Env knobs:
#   INIT_CKPT  - mecka checkpoint to initialize from (weights only)
#   NORM_PATH  - precomputed abc-eva norm_stats dir
#   CKPT_PATH  - resume THIS finetune from its own Lightning checkpoint
#   LR         - optimizer lr (default 5e-5)

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

INIT_CKPT="${INIT_CKPT-/storage/project/r-dxu345-0/agao81/EgoVerse/logs/pi05_mecka_all_6d/pi05_mecka_all_6d_bs64x8_constlr_wrist6d_2026-07-11_06-58-33/checkpoints/epoch_epoch=849.ckpt}"
NORM_PATH="${NORM_PATH-/storage/project/r-dxu345-0/agao81/norm_stats/abc_eva_6d/norm_stats}"
CKPT_PATH="${CKPT_PATH-}"
LR="${LR-5e-5}"

if [[ ! -f "$NORM_PATH/norm_stats.json" ]]; then
  echo "ERROR: precomputed norm stats not found: $NORM_PATH/norm_stats.json" >&2
  echo "Run egomimic/scripts/norm_abc_eva_6d.sbatch first." >&2
  exit 1
fi
if [[ ! -f "$INIT_CKPT" ]]; then
  echo "ERROR: init checkpoint not found: $INIT_CKPT" >&2
  exit 1
fi

ARGS=(
  --config-name=train_zarr_cartesian_pi
  data=abc_eva_pi_6d
  model=pi0.5_ft_abc_eva_6d
  evaluator=eval_pi_wristframe_6d
  +evaluator@train_viz_evaluator=train_viz_pi_wristframe_6d
  launch_params.gpus_per_node=8
  "model.optimizer.lr=$LR"
  "norm_stats.precomputed_norm_path=$NORM_PATH"
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch
  callbacks.model_checkpoint.every_n_epochs=50
  name=pi05_abc_eva_ft
  description=pi05_abc_eva_ft_from_mecka849_bs64x8_wrist6d
)
# Weights-only finetune init unless resuming this run's own checkpoint.
if [[ -n "$CKPT_PATH" ]]; then
  ARGS+=("ckpt_path='$CKPT_PATH'")
else
  ARGS+=("+init_weights_ckpt='$INIT_CKPT'")
fi

echo "[train] srun python -u egomimic/trainHydra.py ${ARGS[*]}"
srun "$PY" -u egomimic/trainHydra.py "${ARGS[@]}"
