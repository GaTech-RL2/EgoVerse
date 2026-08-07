#!/bin/bash
# Full DECOUPLED-action (cut-action-input) 2D obs+action policy training on the
# cleaned, perfect-replay data. No val (evaluator incompat with spatial policy
# val for now); checkpoints via every_n_epochs (monitor=None). Eval offline after.
set -u
export COLUMNS=250
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
JID=$(salloc -A rl2-lab -p rl2-lab --time=4:00:00 --gres=gpu:a40:1 --cpus-per-task=12 --mem=64G --no-shell 2>&1 | grep -oP 'allocation \K[0-9]+')
echo "alloc=$JID"
DATA=${DATA:-/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean}
srun --jobid=$JID --export=ALL .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=dfot_cutaction description=cutaction_v1 mode=train \
  data=tsimulation_full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_dataloader_params.pushshapes_sim.batch_size=1 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=1 \
  data.train_dataloader_params.pushshapes_sim.num_workers=12 \
  model=dfot_pushshapes_image_spatial_policy \
  +model.robomimic_model.outer_stage.decouple_action_noise=true \
  model.scheduler.max_steps=19200 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=30 \
  trainer=debug trainer.max_epochs=120 trainer.min_epochs=120 \
  trainer.limit_train_batches=160 trainer.limit_val_batches=1 \
  trainer.check_val_every_n_epoch=999 \
  trainer.precision=16-mixed +trainer.gradient_clip_val=1.0 \
  logger=csv_wandb > /tmp/o_train_cutaction.txt 2>&1
echo "TRAIN rc=$?  $(tail -2 /tmp/o_train_cutaction.txt)"
R=$(ls -dt logs/dfot_cutaction/* 2>/dev/null | head -1)
ls -la $R/checkpoints/ 2>/dev/null
cp $R/checkpoints/last.ckpt external_ckpts/cutaction_v1_last.ckpt 2>/dev/null && echo COPIED_last
