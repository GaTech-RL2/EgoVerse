#!/bin/bash
# Smoke for the larger obs-as-input H-Net on circle_750 — 2 epochs × 2 batches.
# Catches shape mismatches from the 5-conv image encoder + the larger T8/d=256
# trunk before committing to a 240ep run.
set -euxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse7
VENV=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse7/.venv
source "$VENV/bin/activate"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH=.
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export MUJOCO_GL=egl

TS=$(date +%Y-%m-%d_%H-%M-%S)
DESC="hnet_obs_ar_large_smoke_750_${TS}"

"$VENV/bin/python" -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_smoke \
  description="${DESC}" \
  mode=train \
  data=tsimulation/tsimulation \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/storage/home/hcoda1/4/paphiwetsa3/r-dxu345-0/datasets/pushT/circle_750/circle \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/storage/home/hcoda1/4/paphiwetsa3/r-dxu345-0/datasets/pushT/circle_750/circle \
  model=hnet/pushshapes_obs_ar_large \
  norm_stats.sample_frac=0.1 \
  evaluator=eval_hnet_full \
  callbacks=checkpoints \
  trainer=debug \
  trainer.max_epochs=2 \
  trainer.min_epochs=2 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=2 \
  trainer.profiler=null \
  logger=csv 2>&1
