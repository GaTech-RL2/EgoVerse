#!/bin/bash
# Runs INSIDE a single srun step: pick the emptiest GPU, bind to it, run the smoke.
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=${PACK_COLLATE_MAX_TOTAL_FRAMES:-3200}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
echo "=== GPUs visible in THIS step ==="
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
# pick index with most free memory
BEST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$BEST
echo "=== bound CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="
MODEL=$1; INIT_CKPT=$2; NORM=$3; NAME=$4; BS=$5; shift 5
EXTRA="$@"
python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=${NAME} description=smoke mode=train data=tsimulation model=${MODEL} \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=1 \
  trainer=debug trainer.precision=32 trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=8 trainer.limit_val_batches=2 trainer.check_val_every_n_epoch=1000 trainer.num_sanity_val_steps=0 \
  trainer.profiler=null logger=csv \
  data.train_dataloader_params.pushshapes_sim.batch_size=${BS} \
  data.valid_dataloader_params.pushshapes_sim.batch_size=${BS} \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=$NORM \
  model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  model.scheduler.max_steps=90000 model.scheduler.warmup_steps=4500 \
  model.scheduler.warmup_start_factor=0.1 model.scheduler.eta_min=1.0e-6 \
  $EXTRA \
  +init_ckpt="'$INIT_CKPT'"
echo "SMOKE_EXIT=$?"
