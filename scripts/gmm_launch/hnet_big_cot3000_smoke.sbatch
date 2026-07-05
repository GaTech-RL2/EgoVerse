#!/bin/bash
#SBATCH --job-name=hnet_big_cot3000_smoke
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=embers
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=0:40:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/hnet_big_cot3000_smoke_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
srun .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_big_cot3000_smoke description=hnet_big_cot3000_smoke \
  mode=train \
  data=gmm_cotrain model=hnet_cotrain_gmm_obs_big evaluator=gmm_eval_cotrain \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/circle_3000 \
  data.train_datasets.pushshapes_sim_small_circle.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/small_circle_3000 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/circle_3000 \
  data.valid_datasets.pushshapes_sim_small_circle.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/small_circle_3000 \
  norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=null \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=2 \
  trainer.max_epochs=2 trainer.min_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=2 \
  logger=csv
