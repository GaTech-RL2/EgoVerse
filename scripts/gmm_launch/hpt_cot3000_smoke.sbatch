#!/bin/bash
#SBATCH --job-name=hpt_cot3000_smoke
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=embers
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=0:40:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/hpt_cot3000_smoke_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
srun .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hpt_cot3000_smoke description=hpt_cot3000_smoke \
  mode=train \
  data=tsimulation_hpt_cot3000 model=hpt_cotrain_pushshapes_flow_shared_head evaluator=eval_hpt_standard trainer=ddp_pi \
  norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=null norm_stats.sample_frac=0.01 trainer.precision=32 \
  trainer.max_epochs=2 trainer.min_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.check_val_every_n_epoch=2 \
  callbacks.model_checkpoint.every_n_epochs=2 \
  logger=csv
