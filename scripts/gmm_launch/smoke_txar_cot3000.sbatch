#!/bin/bash
#SBATCH --job-name=smoke_txar_cot3000
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=embers
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=0:40:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/smoke_txar_cot3000_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl

echo "############## SMOKE: TransformerAR (WindowedBC) 2-domain cotrain circle_3000 + small_circle_3000 ##############"
.venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=smoke_txar_cot3000 description=txar_cot3000 mode=train \
  data=gmm_cotrain model=bc_rnn_pushshapes_paperexact_tx evaluator=eval_hnet_sim \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/circle_3000 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/circle_3000 \
  data.train_datasets.pushshapes_sim_small_circle.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/small_circle_3000 \
  data.valid_datasets.pushshapes_sim_small_circle.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/small_circle_3000 \
  +model.robomimic_model.obs_stride=8 \
  +model.robomimic_model.chunk_len=8 \
  +model.robomimic_model.gmm_head.chunk_len=8 \
  model.robomimic_model.rnn_horizon=128 \
  model.robomimic_model.max_windows_per_batch=32 \
  model.robomimic_model.pad_mode=zero_masked \
  'model.robomimic_model.domains=[pushshapes_sim,pushshapes_sim_small_circle]' \
  +model.robomimic_model.ac_keys.pushshapes_sim_small_circle=actions \
  trainer.precision=32 \
  norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=null \
  trainer.max_epochs=2 trainer.min_epochs=1 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=2 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=2 \
  logger=csv
echo "EXIT_TXAR_COT3000=$?"
echo "############## SMOKE DONE ##############"
