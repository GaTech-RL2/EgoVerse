#!/bin/bash
set -euxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse7
mkdir -p logs/sbatch
source .venv/bin/activate
export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"
export MUJOCO_GL=egl
export PYTHONPATH=.
.venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hpt_variants \
  description=hpt_240ep_b32_lr1p5e5_reg_l40s \
  mode=train \
  data=tsimulation_hpt \
  data.train_dataloader_params.pushshapes_sim.batch_size=32 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=32 \
  model=hpt_pushshapes_circle_regression \
  +norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse7/logs/hpt_variants/hpt_120ep_b512_lr1e5_keymapfix_pace_2026-05-19_02-10-03/norm_stats \
  model.optimizer.lr=1.5e-5 \
  '~model.scheduler' \
  +model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  +model.scheduler._partial_=true \
  +model.scheduler.max_steps=3840 \
  +model.scheduler.warmup_steps=48 \
  +model.scheduler.warmup_start_factor=0.1 \
  +model.scheduler.eta_min=2.5e-6 \
  evaluator=eval_hpt_standard \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=60 \
  trainer=debug \
  trainer.max_epochs=240 trainer.min_epochs=240 \
  trainer.limit_train_batches=16 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=60 \
  trainer.profiler=null \
  logger=csv_wandb
