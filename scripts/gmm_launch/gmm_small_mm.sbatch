#!/bin/bash
#SBATCH --job-name=gmm_small_mm
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/gmm_small_mm_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
srun .venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=gmm_small_mm description=gmm_small_mm_200k mode=train \
  data=gmm_smallcircle model=hnet_gmm_obs_small_circle evaluator=gmm_eval_smallcircle \
  +model.robomimic_model.obs_stride=8 \
  norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=null \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=500 \
  trainer.max_epochs=2000 trainer.min_epochs=1 \
  trainer.limit_train_batches=100 trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=500 logger=csv trainer.precision=32 ++evaluator.evals.1.env_kwargs.pusher_shape=circle_small
