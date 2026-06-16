#!/bin/bash
#SBATCH --job-name=s1c1_smoke
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=180G
#SBATCH --time=0:40:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/s1c1_smoke_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
srun .venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=s1c1_smoke description=s1c1_smoke_200k mode=train \
  data=gmm_circle model=hnet_gmm_obs_circle evaluator=gmm_eval_circle \
  +model.robomimic_model.obs_stride=1 \
  norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=null \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=2 \
  trainer.max_epochs=2 trainer.min_epochs=1 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=2 logger=csv trainer.precision=32 model.robomimic_model.outer_stage.gmm_head.chunk_len=1 model.robomimic_model.outer_stage.hnet.stages.0.target_compression_ratio=6.0 model.robomimic_model.outer_stage.hnet.stages.1.target_compression_ratio=6.0
