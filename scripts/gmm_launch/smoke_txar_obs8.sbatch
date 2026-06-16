#!/bin/bash
#SBATCH --job-name=smoke_txar8
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=0:25:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/smoke_txar8_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl

echo "############## SMOKE: TransformerAR obs_stride=8 chunk_len=8 (WindowedBC, no Fix A) ##############"
.venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=smoke_txar8 description=txar \
  data=tsimulation_nc3 model=bc_rnn_pushshapes_paperexact_tx evaluator=eval_hnet_sim \
  +model.robomimic_model.obs_stride=8 \
  +model.robomimic_model.chunk_len=8 \
  +model.robomimic_model.gmm_head.chunk_len=8 \
  model.robomimic_model.rnn_horizon=128 \
  model.robomimic_model.max_windows_per_batch=32 \
  model.robomimic_model.pad_mode=zero_masked \
  trainer.precision=32 \
  norm_stats.norm_mode=minmax norm_stats.sample_frac=0.1 \
  trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=8 trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=2 \
  logger=csv
echo "EXIT_TXAR=$?"
echo "############## SMOKE DONE ##############"
