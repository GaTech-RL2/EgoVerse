#!/bin/bash
#SBATCH -A rl2-lab
#SBATCH --partition=rl2-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --job-name=hpt_bench_inmem
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bench/inmem_reg_fp32_%j.out

set -x
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate

DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3

# Short benchmark: 5 epochs, in-memory fast loader, regression fp32.
# Validates the full pipeline end-to-end (preload -> train -> no crash) and
# lets us read it/s -> s/epoch from the progress log to extrapolate to 500.
python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=bench \
  description=inmem_reg_fp32_bench \
  mode=train \
  model=hpt_pushshapes_circle_regression \
  data=tsimulation_hpt_fast \
  +data_schematic=hpt \
  ~evaluator \
  launch_params.gpus_per_node=1 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_dataloader_params.pushshapes_sim.batch_size=128 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=128 \
  trainer.max_epochs=5 \
  trainer.precision=32 \
  trainer.strategy=ddp_find_unused_parameters_true \
  trainer.check_val_every_n_epoch=100 \
  trainer.limit_val_batches=2 \
  norm_stats.sample_frac=0.1 \
  model.optimizer.lr=4e-4
