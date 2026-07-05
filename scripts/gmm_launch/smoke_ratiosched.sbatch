#!/bin/bash
#SBATCH --job-name=smoke_ratiosched
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=0:20:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/smoke_ratiosched_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
CIRCLE_NORM=logs/gmm_circle/gmm_circle_200k_2026-06-08_12-19-23/norm_stats/norm_stats.json

# 2 epochs, switch_fraction=0.5 -> switch_epoch=round(0.5*2)=1, so epoch 0 = ON
# (0.03), epoch 1 = OFF (0.0). Verifies the callback finds the ChunkerStages
# and flips ratio_loss_weight. check_val high so no val (callback-only smoke).
echo "############## SMOKE: ratio-loss scheduler (circle obs_stride=8) ##############"
.venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=smoke_ratiosched description=circle \
  data=gmm_circle model=hnet_gmm_obs_circle evaluator=gmm_eval_circle \
  +model.robomimic_model.obs_stride=8 \
  norm_stats.precomputed_norm_path=$CIRCLE_NORM \
  callbacks=[checkpoints,ratio_loss_scheduler] \
  trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=8 trainer.limit_val_batches=1 \
  trainer.check_val_every_n_epoch=10 \
  logger=csv
echo "EXIT_RATIOSCHED=$?"
echo "############## SMOKE DONE ##############"
