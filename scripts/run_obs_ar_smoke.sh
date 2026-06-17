#!/bin/bash
# 2-epoch smoke for the obs-as-input H-Net variant. Runs on whichever interactive
# alloc the caller srun's into.
set -euxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse7
VENV=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse7/.venv
source "$VENV/bin/activate"
# Belt-and-suspenders: unset any leaked sibling VIRTUAL_ENV state.
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
unset PYTHONPATH PYTHONHOME 2>/dev/null || true
export PYTHONPATH=.
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200

# Use the absolute python so we can't accidentally pick up a sibling venv.
echo "==using=="; which python; "$VENV/bin/python" -c "import sys; print(sys.prefix); print(sys.executable)"

"$VENV/bin/python" -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_smoke \
  description=hnet_obs_ar_smoke \
  mode=train \
  data=tsimulation/tsimulation \
  model=hnet/pushshapes_obs_ar \
  norm_stats.precomputed_norm_path=logs/hnet_variants/hnet_baseline_pace_l40s_2026-05-19_04-15-39_2026-05-19_04-15-55/norm_stats \
  evaluator=hnet/full \
  callbacks=checkpoints \
  trainer=debug \
  trainer.max_epochs=2 \
  trainer.min_epochs=2 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=2 \
  logger=csv 2>&1
