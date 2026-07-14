#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=smoke_abc_eva
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/smoke_abc_eva_%j.log

# End-to-end eva smoke for the abc finetune path: 16-episode subset,
# trainer=debug (4 epochs, val@2, 2 train/3 val batches), viz forced at
# epoch 3 -> exercises init_weights_ckpt (mecka epoch-849), the mixed-
# resolution collate resize, eva forward/backward, all new val metrics
# (incl. RKL M=4), the train_viz head, and the eva wristframe revert +
# video render.

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

INIT_CKPT=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/pi05_mecka_all_6d/pi05_mecka_all_6d_bs64x8_constlr_wrist6d_2026-07-11_06-58-33/checkpoints/epoch_epoch=849.ckpt

"$PY" -u egomimic/trainHydra.py \
  --config-name=train_zarr_cartesian_pi \
  data=abc_eva_pi_6d \
  model=pi0.5_ft_abc_eva_6d \
  evaluator=eval_pi_wristframe_6d \
  +evaluator@train_viz_evaluator=train_viz_pi_wristframe_6d \
  trainer=debug \
  launch_params.gpus_per_node=1 \
  evaluator.viz_every_n_epochs=4 \
  "++data.train_datasets.eva_bimanual.resolver.debug=16" \
  "data.train_datasets.eva_bimanual.valid_ratio=0.25" \
  "norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/agao81/norm_stats/abc_eva_6d/norm_stats" \
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch \
  "+init_weights_ckpt='$INIT_CKPT'" \
  name=smoke_abc_eva \
  description=smoke_abc_eva_ft_path
