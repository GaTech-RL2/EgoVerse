#!/bin/bash
#SBATCH -A rl2-lab
#SBATCH --partition=rl2-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=1:00:00
#SBATCH --job-name=hpt_bench_4gpu
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bench/inmem_reg_fp32_4gpu_%j.out

set -x
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate

export PYTHONUNBUFFERED=1
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3

# 4-GPU DDP validation. srun launches one task per GPU so Lightning's
# SLURMEnvironment maps ranks correctly (avoids the "1/2 clients joined" hang).
# limit_train_batches=25 (=100/4) keeps per-epoch DATA equal to the 1-GPU
# baseline while splitting it across 4 ranks -> ~4x wall-clock speedup.
srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=bench \
  description=inmem_reg_fp32_4gpu_bench \
  mode=train \
  model=hpt_pushshapes_circle_regression \
  data=tsimulation_hpt_fast \
  +data_schematic=hpt \
  ~evaluator \
  launch_params.gpus_per_node=4 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_dataloader_params.pushshapes_sim.batch_size=128 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=128 \
  data.train_dataloader_params.pushshapes_sim.num_workers=7 \
  data.valid_dataloader_params.pushshapes_sim.num_workers=7 \
  trainer.max_epochs=5 \
  trainer.precision=32 \
  trainer.devices=4 \
  trainer.strategy=ddp_find_unused_parameters_true \
  trainer.limit_train_batches=25 \
  trainer.check_val_every_n_epoch=100 \
  trainer.limit_val_batches=2 \
  norm_stats.sample_frac=0.1 \
  model.optimizer.lr=4e-4
