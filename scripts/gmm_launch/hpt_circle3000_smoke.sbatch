#!/bin/bash
#SBATCH --job-name=hpt_circle3000_smoke
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=embers
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=0:40:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/hpt_circle3000_smoke_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
srun .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hpt_circle3000_smoke description=hpt_circle3000_smoke \
  mode=train \
  data=tsimulation_hpt_nc3 model=hpt_pushshapes_circle evaluator=eval_hpt_standard trainer=ddp_pi \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/circle_3000 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/circle_3000 \
  norm_stats.precomputed_norm_path=null \
  trainer.max_epochs=2 trainer.min_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.check_val_every_n_epoch=2 \
  callbacks.model_checkpoint.every_n_epochs=2 \
  logger=csv
