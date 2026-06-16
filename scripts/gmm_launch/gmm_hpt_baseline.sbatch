#!/bin/bash
#SBATCH --job-name=gmm_hpt_baseline
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250G
#SBATCH --time=3-00:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/gmm_hpt_baseline_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
srun .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=gmm_hpt_baseline description=gmm_hpt_baseline_200k \
  mode=train \
  data=tsimulation_hpt_nc3 model=hpt_pushshapes_circle evaluator=eval_hpt_standard trainer=ddp_pi \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=500 \
  trainer.max_epochs=2000 trainer.min_epochs=1 \
  trainer.limit_train_batches=100 trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=500 \
  logger=csv
