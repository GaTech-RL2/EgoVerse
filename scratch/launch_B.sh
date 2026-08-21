#!/bin/bash
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
export COLUMNS=250 PYTHONPATH=. SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun
$SRUN --jobid=3283274 --overlap .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian name=pixelpol_B description=B_regress_v1 mode=train \
  data=tsimulation_full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean \
  data.train_dataloader_params.pushshapes_sim.batch_size=2 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=2 \
  model=dfot_pushshapes_pixel_regress \
  model.scheduler.max_steps=24000 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=25 ++callbacks.model_checkpoint.save_on_train_epoch_end=true \
  trainer=debug trainer.max_epochs=300 trainer.min_epochs=300 \
  trainer.limit_train_batches=80 trainer.limit_val_batches=1 trainer.check_val_every_n_epoch=999 \
  logger=csv_wandb > /tmp/o_pixelpol_B.txt 2>&1
echo "B_DONE rc=$?"
