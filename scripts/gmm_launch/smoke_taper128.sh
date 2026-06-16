#!/bin/bash
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
DS=/coc/flash7/paphiwetsa3/datasets
.venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=smoke_taper128_c8r2 description=smoke mode=train \
  data=gmm_cotrain model=hnet_cotrain_gmm_obs_big_taper128_c8r2 evaluator=gmm_eval_cotrain \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DS/circle_3000 \
  data.train_datasets.pushshapes_sim_small_circle.resolver.folder_path=$DS/small_circle_3000 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS/circle_3000 \
  data.valid_datasets.pushshapes_sim_small_circle.resolver.folder_path=$DS/small_circle_3000 \
  norm_stats.norm_mode=minmax norm_stats.sample_frac=0.02 trainer.precision=32 \
  callbacks=checkpoints \
  trainer.max_epochs=2 trainer.min_epochs=1 \
  trainer.limit_train_batches=4 trainer.limit_val_batches=1 trainer.check_val_every_n_epoch=999 \
  logger=csv
echo "SMOKE_EXIT=$?"
