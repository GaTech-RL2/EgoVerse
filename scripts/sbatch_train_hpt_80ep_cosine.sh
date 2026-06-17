#!/bin/bash
#SBATCH --job-name=hpt-80ep-cosine
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse/logs/sbatch/hpt_80ep_cosine_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse/logs/sbatch/hpt_80ep_cosine_%j.err

# HPT proper retrain (apples-to-apples vs H-Net): 50 epochs, lr=4e-5 with
# warmup+cosine matching the H-Net schedule. HPT uses limit_train_batches=16
# (vs H-Net's 8), so total steps = 50 x 16 = 800. Warmup 3 epochs (48 steps),
# eta_min 4e-6 (same 10% floor as H-Net).

set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse
mkdir -p logs/sbatch

hostname
nvidia-smi || true
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}  SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"

export PYTHONPATH=.
srun --kill-on-bad-exit=1 .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hpt_variants \
  description=hpt_80ep_cosine_lr4e5 \
  mode=train \
  data=tsimulation_hpt \
  model=hpt/pushshapes_circle \
  model.optimizer.lr=4e-5 \
  '~model.scheduler' \
  +model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  +model.scheduler._partial_=true \
  +model.scheduler.max_steps=1280 \
  +model.scheduler.warmup_steps=48 \
  +model.scheduler.warmup_start_factor=0.1 \
  +model.scheduler.eta_min=4.0e-6 \
  evaluator=eval_hpt_standard \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=10 \
  trainer=debug \
  trainer.max_epochs=80 \
  trainer.min_epochs=80 \
  trainer.limit_train_batches=16 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=10 \
  trainer.profiler=null \
  logger=csv_wandb
