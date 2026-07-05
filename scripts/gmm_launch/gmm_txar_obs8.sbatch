#!/bin/bash
#SBATCH --job-name=gmm_txar_obs8
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250G
#SBATCH --time=3-00:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/gmm_txar_obs8_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
srun .venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=gmm_txar_obs8 description=gmm_txar_obs8_200k mode=train \
  data=tsimulation_nc3 model=bc_rnn_pushshapes_paperexact_tx evaluator=eval_hnet_sim \
  +model.robomimic_model.obs_stride=8 +model.robomimic_model.chunk_len=8 \
  +model.robomimic_model.gmm_head.chunk_len=8 \
  model.robomimic_model.rnn_horizon=128 model.robomimic_model.max_windows_per_batch=32 \
  model.robomimic_model.pad_mode=zero_masked \
  trainer.precision=32 norm_stats.norm_mode=minmax \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=500 \
  trainer.max_epochs=2000 trainer.min_epochs=1 \
  trainer.limit_train_batches=100 trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=500 logger=csv
