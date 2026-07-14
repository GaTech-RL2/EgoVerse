#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_aria_grip
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/eval_aria_grip_%j.log

# Eval-only on the aria fold grip data: mecka epoch-849 weights (same init
# the smoke evaluated), full 5% val split, 400 batches (vs the smoke's 3) —
# statistical check on whether the smoke's metrics were an outlier.
#
# Env knob: CKPT - checkpoint to evaluate (default: mecka epoch-849)

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CKPT="${CKPT-/storage/project/r-dxu345-0/agao81/EgoVerse/logs/pi05_mecka_all_6d/pi05_mecka_all_6d_bs64x8_constlr_wrist6d_2026-07-11_06-58-33/checkpoints/epoch_epoch=849.ckpt}"

"$PY" -u egomimic/trainHydra.py \
  --config-name=train_zarr_cartesian_pi \
  data=aria_fold_pi_6d_grip \
  model=pi0.5_ft_aria_fold_grip \
  evaluator=eval_pi_wristframe_6d_grip \
  launch_params.gpus_per_node=1 \
  ++mode=eval \
  "ckpt_path='$CKPT'" \
  "norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/agao81/norm_stats/aria_fold_grip_6d/norm_stats" \
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch \
  name=eval_aria_grip \
  description=eval_only_aria_fold_grip_400batches
