#!/bin/bash
#SBATCH --job-name=hnet-adaln-qreg-80ep
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse/logs/sbatch/hnet_adaln_qreg_80ep_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse/logs/sbatch/hnet_adaln_qreg_80ep_%j.err

# H-Net AdaLN with eval fixes + 1/4 chunk regularizer (0.03 -> 0.0075).
# Eval fixes applied:
#   - viz_gt_preds now emits 1 frame/timestep (no chunk dwell) -> model
#     panel sync'd with PCA + boundary strip panels.
#   - max_videos=2 across all composite sub-evals + sim eval.
#   - Sim eval writes to videos/sim/ (no path collision).
# 1/4 regularizer is to test whether chunk too short and consecutive
# boundary degenerate mode improves at lower regularization pressure.

set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse
mkdir -p logs/sbatch

hostname
nvidia-smi || true

export PYTHONPATH=.
srun --kill-on-bad-exit=1 .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_variants \
  description=hnet_adaln_80ep_cosine_lr4e5_qreg \
  mode=train \
  data=tsimulation \
  model=hnet_pushshapes \
  model.scheduler.max_steps=640 \
  model.robomimic_model.hnet.target_compression_ratio=8.0 \
  model.robomimic_model.hnet.ratio_loss_weight=0.0075 \
  evaluator=eval_hnet_full \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=20 \
  trainer=debug \
  trainer.max_epochs=80 \
  trainer.min_epochs=80 \
  trainer.limit_train_batches=8 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=20 \
  trainer.profiler=null \
  logger=csv_wandb
