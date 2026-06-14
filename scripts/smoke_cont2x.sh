#!/bin/bash
# Smoke one cont2x run: 2 epochs, csv logger, WANDB off, eval skipped, _smoke name.
# Usage: smoke_cont2x.sh <MODEL> <INIT_CKPT> <NORM> <SMOKENAME> [EXTRA_ARGS...]
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export WANDB_MODE=disabled
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

MODEL=$1; INIT_CKPT=$2; NORM=$3; NAME=$4; shift 4
EXTRA_ARGS="$@"
LTB=50
MAXSTEPS_SCHED=$(( 1800 * LTB ))   # keep the real fresh-cosine horizon so the LR we
                                   # observe at step ~20000+ matches what the real run sees

python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=smoke_cont2x mode=train data=tsimulation \
  model=${MODEL} \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=1 \
  trainer=debug trainer.precision=32 trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=2 trainer.check_val_every_n_epoch=1000 \
  trainer.num_sanity_val_steps=0 \
  trainer.profiler=null logger=csv \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=$NORM \
  model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  model.scheduler.max_steps=${MAXSTEPS_SCHED} \
  model.scheduler.warmup_steps=4500 model.scheduler.warmup_start_factor=0.1 model.scheduler.eta_min=1.0e-6 \
  $EXTRA_ARGS \
  +init_ckpt="'$INIT_CKPT'"
echo "SMOKE_EXIT=$?"
