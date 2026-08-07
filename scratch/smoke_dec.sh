#!/bin/bash
set -u
export COLUMNS=250
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
JID=$(salloc -A rl2-lab -p rl2-lab --time=1:00:00 --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --no-shell 2>&1 | grep -oP 'allocation \K[0-9]+')
echo "alloc=$JID"
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3
srun --jobid=$JID --export=ALL .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=smoke_dec description=smoke mode=train \
  data=tsimulation_full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_dataloader_params.pushshapes_sim.batch_size=1 \
  model=dfot_pushshapes_image_spatial_policy \
  +model.robomimic_model.outer_stage.decouple_action_noise=true \
  trainer=debug trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=4 trainer.limit_val_batches=1 \
  trainer.check_val_every_n_epoch=999 \
  norm_stats.sample_frac=0.1 logger=csv > /tmp/o_smokedec.txt 2>&1
echo "SMOKE rc=$?"
grep -iE "error|exception|Traceback|OutOfMemory|NotImplemented|Train/|Epoch|action_loss|Starting training|decouple" /tmp/o_smokedec.txt | tail -12
scancel $JID 2>/dev/null
