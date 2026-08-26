#!/bin/bash
set -u
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
export PYTHONPATH=. SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean
JID=$1
$SRUN --jobid=$JID --overlap .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=debug description=A32smoke mode=train \
  data=tsimulation_full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_datasets.pushshapes_sim.chunking=sequential \
  data.valid_datasets.pushshapes_sim.chunking=sequential \
  data.train_datasets.pushshapes_sim.max_seq_len=33 \
  data.valid_datasets.pushshapes_sim.max_seq_len=33 \
  data.train_datasets.pushshapes_sim.min_seq_len=33 \
  data.valid_datasets.pushshapes_sim.min_seq_len=33 \
  data.train_dataloader_params.pushshapes_sim.batch_size=32 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=32 \
  model=dfot/pixel_policy \
  model.robomimic_model.action_horizon=33 \
  model.scheduler.max_steps=4 \
  norm_stats.sample_frac=0.1 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=2 \
  ++callbacks.model_checkpoint.save_on_train_epoch_end=true \
  trainer=debug trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=1 \
  trainer.check_val_every_n_epoch=2 \
  logger=csv > /tmp/o_A32smoke.txt 2>&1
echo "A32smoke_rc=$?"
tail -6 /tmp/o_A32smoke.txt
