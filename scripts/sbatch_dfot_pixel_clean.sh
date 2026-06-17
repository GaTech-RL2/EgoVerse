#!/bin/bash
#SBATCH --job-name=dfot-clean
#SBATCH -A hoffman-lab
#SBATCH -p overcap
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=short
#SBATCH --no-requeue
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/logs/sbatch/dfot_clean_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/logs/sbatch/dfot_clean_%j.err

set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
mkdir -p logs/sbatch

source .venv/bin/activate
export PYTHONPATH=.

srun --kill-on-bad-exit=1 .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=dfot_pixel_clean description=pixel_clean_fused_min_snr \
  mode=train \
  data=video_clips \
  model=dfot/pixel_video \
  evaluator=eval_dfot_pixel \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=100 \
  trainer=debug \
  trainer.max_epochs=400 \
  trainer.min_epochs=400 \
  trainer.limit_train_batches=80 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=50 \
  trainer.profiler=simple \
  trainer.precision=16-mixed \
  +trainer.gradient_clip_val=1.0 \
  logger=csv_wandb
