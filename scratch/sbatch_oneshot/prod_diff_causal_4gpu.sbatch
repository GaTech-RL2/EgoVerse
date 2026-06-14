#!/bin/bash
#SBATCH -A overcap
#SBATCH --partition=overcap
#SBATCH --gres=gpu:a40:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --job-name=hpt_diff_causal
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_inmem/reg_fp32_4xa40_%j.out

set -x
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate

export PYTHONUNBUFFERED=1
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3

# Regression HPT, fp32, 4xA40 DDP. limit_train_batches=25 keeps per-epoch data
# equal to the 1-GPU baseline (100 batches) split over 4 ranks. --requeue +
# SIGUSR1 lets trainHydra checkpoint and auto-resume from last.ckpt if the 4h
# QOS limit is hit before 500 epochs. Default callbacks save every 100 epochs.
srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hpt_inmem \
  description=diff_causal_4xa40 \
  mode=train \
  model=hpt_pushshapes_circle \
  data=tsimulation_hpt_causal \
  +data_schematic=hpt \
  ~evaluator \
  launch_params.gpus_per_node=4 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_dataloader_params.pushshapes_sim.batch_size=128 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=128 \
  data.train_dataloader_params.pushshapes_sim.num_workers=7 \
  data.valid_dataloader_params.pushshapes_sim.num_workers=7 \
  trainer.max_epochs=500 \
  trainer.precision=32 \
  trainer.devices=4 \
  trainer.strategy=ddp_find_unused_parameters_true \
  trainer.limit_train_batches=25 \
  trainer.check_val_every_n_epoch=100 \
  trainer.limit_val_batches=10 \
  norm_stats.sample_frac=0.1 \
  model.optimizer.lr=1e-4
