#!/bin/bash
#SBATCH --job-name=resmoke_hnet_big_circle3000
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250G
#SBATCH --time=0:40:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/resmoke_hnet_big_circle3000_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
srun .venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=hnet_big_circle3000 description=hnet_big_circle3000_200k mode=train \
  data=gmm_circle model=hnet_gmm_obs_circle_big evaluator=gmm_eval_circle \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/circle_3000 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/circle_3000 \
  +model.robomimic_model.obs_stride=8 \
  norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=null norm_stats.sample_frac=0.05 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=2 \
  trainer.max_epochs=2 trainer.min_epochs=1 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=2 logger=csv model.robomimic_model.outer_stage.input_modules.0.obs_encoder.obs_specs.state_agent_obj.input_dim=2 +model.robomimic_model.outer_stage.input_modules.0.obs_encoder.obs_specs.state_agent_obj.input_slice=[0,2] trainer.precision=32 model.scheduler._target_=egomimic.utils.schedulers.constant_scheduler ~model.scheduler.max_steps ~model.scheduler.warmup_steps ~model.scheduler.warmup_start_factor ~model.scheduler.eta_min
