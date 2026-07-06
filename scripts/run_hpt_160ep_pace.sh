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
  description=hpt_160ep_cosine_lr1e5_keymapfix_pace \
  mode=train \
  data=tsimulation_hpt \
  model=hpt_pushshapes_circle \
  model.optimizer.lr=1e-5 \
  '~model.scheduler' \
  +model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  +model.scheduler._partial_=true \
  +model.scheduler.max_steps=2560 \
  +model.scheduler.warmup_steps=48 \
  +model.scheduler.warmup_start_factor=0.1 \
  +model.scheduler.eta_min=1.0e-6 \
  evaluator=eval_hpt_standard \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=40 \
  trainer=debug \
  trainer.max_epochs=160 trainer.min_epochs=160 \
  trainer.limit_train_batches=16 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=40 \
  trainer.profiler=null \
  logger=csv_wandb
