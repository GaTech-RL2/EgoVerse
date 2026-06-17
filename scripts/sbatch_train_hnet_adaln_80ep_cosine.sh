#!/bin/bash
#SBATCH --job-name=hnet-adaln-80ep-cosine
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse/logs/sbatch/hnet_adaln_80ep_cosine_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse/logs/sbatch/hnet_adaln_80ep_cosine_%j.err

# H-Net AdaLN (hnet_pushshapes) — 50 epochs, warmup+cosine LR schedule:
#   start lr 4e-5, warmup 3 epochs (24 steps), cosine -> eta_min 4e-6 by step 400.
# Validation runs eval_hnet_full = val viz + PCA + boundary strip + sim eval
# (sim eval capped at 2 videos via max_videos).

set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse
mkdir -p logs/sbatch

hostname
nvidia-smi || true
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}  SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"

export PACK_COLLATE_MAX_TOTAL_FRAMES=3200  # match max_seq_len; pack_collate drops samples to fit
export PYTHONPATH=.
srun --kill-on-bad-exit=1 .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_variants \
  description=hnet_adaln_80ep_cosine_lr4e5 \
  mode=train \
  data=tsimulation/tsimulation \
  model=hnet/pushshapes \
  model.scheduler.max_steps=640 \
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
