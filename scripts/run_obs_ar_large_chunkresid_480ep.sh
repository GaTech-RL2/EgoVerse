#!/bin/bash
# 480ep (double) larger obs-as-input H-Net with chunker_residual_scheduler.
# Schedule scaled to match: skip-off for first 1920 steps (half), on from 1920+.
# 4 val cadence: check_val_every_n_epoch=120.
set -euxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse7
VENV=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse7/.venv
source "$VENV/bin/activate"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH=.
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export MUJOCO_GL=egl
export WANDB_ENTITY=rl2-group

TS=$(date +%Y-%m-%d_%H-%M-%S)
DESC="hnet_obs_ar_large_chunkresid12_480ep_pace_h200_750_${TS}"

"$VENV/bin/python" -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_variants \
  description="${DESC}" \
  mode=train \
  data=tsimulation/tsimulation \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/storage/home/hcoda1/4/paphiwetsa3/r-dxu345-0/datasets/pushT/circle_750/circle \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/storage/home/hcoda1/4/paphiwetsa3/r-dxu345-0/datasets/pushT/circle_750/circle \
  model=hnet/pushshapes_obs_ar_large \
  model.scheduler.max_steps=3840 \
  evaluator=eval_hnet_full \
  callbacks=ckpt_chunker \
  ++callbacks.chunker_residual_scheduler.schedule="[[0,0.0],[1919,0.0],[1920,1.0]]" \
  ++callbacks.model_checkpoint.every_n_epochs=120 \
  trainer=debug \
  trainer.max_epochs=480 \
  trainer.min_epochs=480 \
  trainer.limit_train_batches=8 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=120 \
  trainer.profiler=null \
  logger=csv_wandb 2>&1
