#!/bin/bash
set -u
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
export PYTHONPATH=. SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean
NORM=logs/pixelpol_A/A_diffuse_v2_2026-05-31_17-27-49/norm_stats/norm_stats.json
CKPT=logs/pixeldec/DEC_v1_2026-06-01_00-11-19/checkpoints/last.ckpt
JID=$1
$SRUN --jobid=$JID --overlap .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=pixeldec description=DEC_v2_long mode=train \
  ckpt_path="$CKPT" \
  data=tsimulation_full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_datasets.pushshapes_sim.chunking=sequential \
  data.valid_datasets.pushshapes_sim.chunking=sequential \
  data.train_datasets.pushshapes_sim.max_seq_len=9 \
  data.valid_datasets.pushshapes_sim.max_seq_len=9 \
  data.train_datasets.pushshapes_sim.min_seq_len=9 \
  data.valid_datasets.pushshapes_sim.min_seq_len=9 \
  data.train_dataloader_params.pushshapes_sim.batch_size=8 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=8 \
  model=dfot/pixel_decoupled \
  model.scheduler.max_steps=24000 \
  norm_stats.precomputed_norm_path=$NORM \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=25 \
  ++callbacks.model_checkpoint.save_on_train_epoch_end=true \
  trainer=debug trainer.max_epochs=300 trainer.min_epochs=300 \
  trainer.limit_train_batches=80 trainer.limit_val_batches=1 \
  trainer.check_val_every_n_epoch=999 \
  logger=csv_wandb > /tmp/o_DEC_v2.txt 2>&1
echo "DEC_v2_DONE rc=$?"
tail -6 /tmp/o_DEC_v2.txt
