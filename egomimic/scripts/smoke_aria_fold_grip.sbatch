#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=smoke_aria_grip
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/smoke_aria_grip_%j.log

# End-to-end smoke for the aria fold keypoint-gripper finetune: 16-episode
# subset, trainer=debug, viz forced at epoch 3 -> exercises the grip keymap
# keys, KeypointsToGripper/InsertGripperChannels (20D actions+proprio), the
# ROBOT converter on human data, init from the mecka checkpoint, all val
# metrics, the train_viz head, and the grip revert (action unpad) + video.

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INIT_CKPT=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/pi05_mecka_all_6d/pi05_mecka_all_6d_bs64x8_constlr_wrist6d_2026-07-11_06-58-33/checkpoints/epoch_epoch=849.ckpt

"$PY" -u egomimic/trainHydra.py \
  --config-name=train_zarr_cartesian_pi \
  data=aria_fold_pi_6d_grip \
  model=pi0.5_ft_aria_fold_grip \
  evaluator=eval_pi_wristframe_6d_grip \
  +evaluator@train_viz_evaluator=train_viz_pi_wristframe_6d_grip \
  trainer=debug \
  launch_params.gpus_per_node=1 \
  evaluator.viz_every_n_epochs=4 \
  "++data.train_datasets.human_bimanual.resolver.debug=16" \
  "data.train_datasets.human_bimanual.valid_ratio=0.25" \
  "norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/agao81/norm_stats/aria_fold_grip_6d/norm_stats" \
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch \
  "+init_weights_ckpt='$INIT_CKPT'" \
  name=smoke_aria_grip \
  description=smoke_aria_fold_grip_path
