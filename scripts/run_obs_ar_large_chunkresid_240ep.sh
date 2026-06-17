#!/bin/bash
# 240ep larger obs-as-input H-Net WITH chunker_residual_scheduler (no attn dropout).
# First half (steps 0-959): residual_scale=0 — chunker can't skip its boundary head.
# Second half (steps 960-1920): residual_scale=1 — full skip path on.
# Dataset: circle_750 (750 episodes, larger than original circle/basic 450).
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
DESC="hnet_obs_ar_large_chunkresid12_240ep_pace_a100_750_${TS}"

"$VENV/bin/python" -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_variants \
  description="${DESC}" \
  mode=train \
  data=tsimulation/tsimulation \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/storage/home/hcoda1/4/paphiwetsa3/r-dxu345-0/datasets/pushT/circle_750/circle \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/storage/home/hcoda1/4/paphiwetsa3/r-dxu345-0/datasets/pushT/circle_750/circle \
  model=hnet/pushshapes_obs_ar_large \
  model.scheduler.max_steps=1920 \
  evaluator=hnet/full \
  callbacks=ckpt_chunker \
  trainer=debug \
  trainer.max_epochs=240 \
  trainer.min_epochs=240 \
  trainer.limit_train_batches=8 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=60 \
  trainer.profiler=null \
  logger=csv_wandb 2>&1
