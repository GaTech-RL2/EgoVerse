#!/bin/bash
#SBATCH --job-name=dfot-200ep-full750
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/sbatch/dfot_200ep_full750_%j.out
#SBATCH --error=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/sbatch/dfot_200ep_full750_%j.err

# DFoT v1 on PushShapes circle 750-episode full dataset
# (pushT/circle_750/circle). Packed training (tsimulation_full.yaml ->
# variable-length episodes with cu_seqlens within-episode attention).
# Per-frame obs naturally aligns with per-token action via DFoT's per-token
# AdaLN.
#
# Schedule: 200 ep x 160 batches = 32000 steps. Warmup-cosine: 480 warmup
# steps (~3 ep), eta_min 4e-6, lr 4e-5. --time set to 12h (measured H200
# throughput ~100 steps/min in packed mode -> ~5.3h actual; 12h leaves
# margin for slower val passes with both evals enabled).
#
# Eval: eval_dfot_full — composite of:
#   * eval_dfot_val (teacher-forced, full-chunk DDIM + staircase-AR
#     overlaid on GT env frames)
#   * eval_dfot_sim (closed-loop PushShapesEnv rollout via
#     DFoT.inference_step in AR mode by default)
# Val every 25 epochs (=8 val passes over 200 ep).

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
  description=dfot_200ep_full750_lr4e5 \
  mode=train \
  data=tsimulation/full \
  model=dfot/pushshapes \
  model.optimizer.lr=4.0e-5 \
  model.scheduler.max_steps=32000 \
  model.scheduler.warmup_steps=480 \
  model.scheduler.warmup_start_factor=0.1 \
  model.scheduler.eta_min=4.0e-6 \
  evaluator=eval_dfot_full \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=25 \
  trainer=debug \
  trainer.max_epochs=200 \
  trainer.min_epochs=200 \
  trainer.limit_train_batches=160 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=25 \
  trainer.profiler=null \
  logger=csv_wandb
