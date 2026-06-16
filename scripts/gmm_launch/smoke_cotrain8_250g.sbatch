#!/bin/bash
#SBATCH --job-name=smoke_cotrain8_250g
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250G
#SBATCH --time=0:30:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/smoke_cotrain8_250g_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "### cotrain obs_stride=8 RE-SMOKE at 250G ###"
.venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=smoke_cotrain8_250g description=cotrain mode=train \
  data=gmm_cotrain model=hnet_cotrain_gmm_obs evaluator=gmm_eval_cotrain \
  +model.robomimic_model.obs_stride=8 \
  norm_stats.precomputed_norm_path=logs/gmm_cotrain/gmm_cotrain_200k_2026-06-08_16-28-40/norm_stats/norm_stats.json \
  trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=8 trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=2 logger=csv
echo "EXIT_COTRAIN8=$?"
