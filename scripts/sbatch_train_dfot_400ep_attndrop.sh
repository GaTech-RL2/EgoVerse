#!/bin/bash
#SBATCH --job-name=dfot-400ep-attndrop
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250G
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/logs/sbatch/dfot_400ep_attndrop_%j.out
#SBATCH --error=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/logs/sbatch/dfot_400ep_attndrop_%j.err

# DFoT 400ep + random per-batch attn dropout from {0.1, 0.5, 0.8, 0.9, 0.95}.
# Motivation: long packed episodes (up to 400 tokens) + i.i.d. per-token noise
# at training give the causal attention ~10-20 near-clean neighbor tokens to
# copy from at any step. Reference DFoT trains on 8-50 token clips so this
# leak surface barely exists. The random_attn_dropout callback hammers the
# attn path across a wide range of dropout rates so the model can't rely on
# any specific clean neighbor.
#
# Scheduler fix vs 200ep run: max_steps=18800 (= 400 * 47 actual steps/epoch).

set -euxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3
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
  description=dfot_400ep_attndrop \
  mode=train \
  data=tsimulation/full \
  model=dfot/pushshapes \
  model.optimizer.lr=4.0e-5 \
  model.scheduler.max_steps=18800 \
  model.scheduler.warmup_steps=480 \
  model.scheduler.warmup_start_factor=0.1 \
  model.scheduler.eta_min=4.0e-6 \
  evaluator=dfot/full \
  callbacks=ckpt_attn_dropout \
  callbacks.model_checkpoint.every_n_epochs=50 \
  trainer=debug \
  trainer.max_epochs=400 \
  trainer.min_epochs=400 \
  trainer.limit_train_batches=160 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=50 \
  trainer.profiler=null \
  logger=csv
