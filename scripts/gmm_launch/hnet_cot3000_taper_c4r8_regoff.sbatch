#!/bin/bash
#SBATCH --job-name=hnet_cot3000_taper_c4r8_regoff
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=350G
#SBATCH --time=3-00:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/hnet_cot3000_taper_c4r8_regoff_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
srun .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_cot3000_taper_c4r8_regoff description=hnet_cot3000_taper_c4r8_regoff_200k \
  mode=train \
  data=gmm_cotrain model=hnet_cotrain_gmm_obs_big_taper_c4r8 evaluator=gmm_eval_cotrain \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/circle_3000_plus_gen \
  data.train_datasets.pushshapes_sim_small_circle.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/small_circle_3000 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/circle_3000_plus_gen \
  data.valid_datasets.pushshapes_sim_small_circle.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/small_circle_3000 \
  norm_stats.norm_mode=minmax trainer.precision=32 norm_stats.precomputed_norm_path=logs/hnet_big_cot3000/hnet_big_cot3000_200k_2026-06-13_16-56-13/norm_stats \
  callbacks=[checkpoints,ratio_loss_scheduler] callbacks.ratio_loss_scheduler.switch_fraction=0.25 callbacks.model_checkpoint.every_n_epochs=500 \
  trainer.max_epochs=2000 trainer.min_epochs=1 \
  trainer.limit_train_batches=100 trainer.limit_val_batches=2 trainer.check_val_every_n_epoch=500 \
  logger=csv
