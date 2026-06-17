#!/bin/bash
#SBATCH --job-name=dfot-400ep-full750
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250G
#SBATCH --time=20:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/sbatch/dfot_400ep_full750_%j.out
#SBATCH --error=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/sbatch/dfot_400ep_full750_%j.err

# DFoT 400ep run on PushShapes circle 750-ep full dataset.
# Fixes scheduler/step mismatch from the 200ep version: dataset yields
# ~47 packed batches/epoch, NOT 160 (limit_train_batches was a CAP). So
# actual total steps = 400 * 47 = 18800. The 200ep version had max_steps
# set to 32000, which left the LR plateaued near peak instead of decaying.
# Here max_steps=18800 matches the real total so cosine descends fully.
#
# 8 val passes over 400ep (every 50 epochs).

set -euxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4
mkdir -p logs/sbatch

hostname
nvidia-smi || true
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}  SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"

source .venv/bin/activate
export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"
export MUJOCO_GL=egl
export PYTHONPATH=.

srun --kill-on-bad-exit=1 .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=dfot_variants \
  description=dfot_400ep_full750_lr4e5_sched_fixed \
  mode=train \
  data=tsimulation/full \
  model=dfot/pushshapes \
  model.optimizer.lr=4.0e-5 \
  model.scheduler.max_steps=18800 \
  model.scheduler.warmup_steps=480 \
  model.scheduler.warmup_start_factor=0.1 \
  model.scheduler.eta_min=4.0e-6 \
  evaluator=eval_dfot_full \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=50 \
  trainer=debug \
  trainer.max_epochs=400 \
  trainer.min_epochs=400 \
  trainer.limit_train_batches=160 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=50 \
  trainer.profiler=null \
  logger=csv_wandb
