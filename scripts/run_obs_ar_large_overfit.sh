#!/bin/bash
# Overfit sanity check for the larger obs-as-input H-Net.
# 200 epochs on 2 fixed batches (limit_train_batches=2). No val — pure
# capacity probe. If train loss doesn't drop near 0, there's a bug (input
# wiring, gradient flow, etc.). Expected runtime: ~5-10 min.
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
DESC="hnet_obs_ar_large_overfit_${TS}"

"$VENV/bin/python" -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_overfit \
  description="${DESC}" \
  mode=train \
  data=tsimulation \
  model=hnet_pushshapes_obs_ar_large \
  norm_stats.precomputed_norm_path=logs/hnet_variants/hnet_baseline_pace_l40s_2026-05-19_04-15-39_2026-05-19_04-15-55/norm_stats \
  evaluator=eval_hnet_full \
  callbacks=checkpoints \
  ++callbacks.model_checkpoint.every_n_epochs=200 \
  trainer=debug \
  trainer.max_epochs=200 \
  trainer.min_epochs=200 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=200 \
  trainer.profiler=null \
  logger=csv 2>&1
