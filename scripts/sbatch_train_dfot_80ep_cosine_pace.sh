#!/bin/bash
#SBATCH --job-name=dfot-80ep-cosine
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250G
#SBATCH --time=4:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/sbatch/dfot_80ep_cosine_%j.out
#SBATCH --error=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/sbatch/dfot_80ep_cosine_%j.err

# DFoT v1 on PushShapes circle/basic. Packed training (tsimulation.yaml ->
# variable-length episodes with cu_seqlens within-episode attention).
# Per-frame obs naturally aligns with per-token action via DFoT's per-token
# AdaLN. Schedule mirrors the H-Net retrain (lr=4e-5, warmup-cosine, 80 ep,
# limit_train_batches=8 -> 640 steps total) for apples-to-apples comparison
# of train loss + closed-loop coverage.
#
# Eval: eval_hnet_sim — closed-loop PushShapesEnv rollout per val episode,
# 4 val batches x 8 episodes = 32 rollouts per check. inference_step uses
# the current chunk-replan (action_horizon=32) path; AR sampler (causal
# staircase) is a separate launch.

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
  description=dfot_80ep_cosine_lr4e5 \
  mode=train \
  data=tsimulation \
  model=dfot_pushshapes \
  model.optimizer.lr=4.0e-5 \
  model.scheduler.max_steps=640 \
  model.scheduler.warmup_steps=24 \
  model.scheduler.warmup_start_factor=0.1 \
  model.scheduler.eta_min=4.0e-6 \
  evaluator=eval_hnet_sim \
  evaluator.limit_val_batches=4 \
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
