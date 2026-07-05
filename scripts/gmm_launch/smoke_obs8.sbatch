#!/bin/bash
#SBATCH --job-name=smoke_obs8
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=0:35:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/smoke_obs8_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl

CIRCLE_NORM=logs/gmm_circle/gmm_circle_200k_2026-06-08_12-19-23/norm_stats/norm_stats.json
COTRAIN_NORM=logs/gmm_cotrain/gmm_cotrain_200k_2026-06-08_16-28-40/norm_stats/norm_stats.json

echo "############## SMOKE A: circle obs_stride=8 (single-domain, audited path) ##############"
.venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=smoke_obs8 description=circle \
  data=gmm_circle model=hnet_gmm_obs_circle evaluator=gmm_eval_circle \
  +model.robomimic_model.obs_stride=8 \
  norm_stats.precomputed_norm_path=$CIRCLE_NORM \
  trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=8 trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=2 \
  logger=csv
echo "EXIT_A=$?"

echo "############## SMOKE B: cotrain obs_stride=8 (PerEmbodimentStage path, FLAGGED) ##############"
.venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=smoke_obs8 description=cotrain \
  data=gmm_cotrain model=hnet_cotrain_gmm_obs evaluator=gmm_eval_cotrain \
  +model.robomimic_model.obs_stride=8 \
  norm_stats.precomputed_norm_path=$COTRAIN_NORM \
  trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=8 trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=2 \
  logger=csv
echo "EXIT_B=$?"
echo "############## SMOKE DONE ##############"
