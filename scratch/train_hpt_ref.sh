#!/bin/bash
# HPT-regression reference on the cleaned perfect-replay data (the ~HPT target + fallback).
set -u
export COLUMNS=250 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
JID=$(salloc -A rl2-lab -p rl2-lab --time=4:00:00 --gres=gpu:a40:1 --cpus-per-task=12 --mem=64G --no-shell 2>&1 | grep -oP 'allocation \K[0-9]+')
echo "alloc=$JID"
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean
srun --jobid=$JID --export=ALL .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hpt_ref description=hpt_regression_v1 mode=train \
  data=tsimulation_hpt \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_dataloader_params.pushshapes_sim.batch_size=32 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=32 \
  model=hpt_pushshapes_circle_regression \
  '~model.scheduler' \
  +model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  +model.scheduler._partial_=true +model.scheduler.max_steps=3840 \
  +model.scheduler.warmup_steps=48 +model.scheduler.warmup_start_factor=0.1 +model.scheduler.eta_min=2.5e-6 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=60 \
  trainer=debug trainer.max_epochs=240 trainer.min_epochs=240 \
  trainer.limit_train_batches=16 trainer.limit_val_batches=1 \
  trainer.check_val_every_n_epoch=999 \
  trainer.precision=16-mixed +trainer.gradient_clip_val=1.0 \
  logger=csv_wandb > /tmp/o_train_hpt.txt 2>&1
echo "HPT rc=$?  $(tail -2 /tmp/o_train_hpt.txt)"
R=$(ls -dt logs/hpt_ref/* 2>/dev/null | head -1)
ls -la $R/checkpoints/ 2>/dev/null
cp $R/checkpoints/last.ckpt external_ckpts/hpt_ref_v1_last.ckpt 2>/dev/null && echo COPIED_last
