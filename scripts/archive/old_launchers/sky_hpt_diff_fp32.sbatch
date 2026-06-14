#!/bin/bash
#SBATCH -A rl2-lab
#SBATCH --partition=rl2-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --job-name=hpt_diff_fp32
#SBATCH -o /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_data_ablation/hpt_diff_fp32_%j.out
#SBATCH -e /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_data_ablation/hpt_diff_fp32_%j.err
set -euxo pipefail
mkdir -p /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_data_ablation
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=.
export MUJOCO_GL=egl
export WANDB_ENTITY=rl2-group
export DATASET_PT=/coc/flash7/paphiwetsa3/datasets/new_circle_3_materialized/dataset.pt

python scripts/fast_dataloader_wrapper.py \
  --config-name=train_zarr_cartesian \
  name=hpt_data_ablation \
  description=diff_fp32_newcircle3_2xA40 \
  mode=train \
  model=hpt_pushshapes_circle \
  data=tsimulation_hpt \
  +data_schematic=hpt \
  ~evaluator \
  ++trainer.strategy=ddp_find_unused_parameters_true \
  launch_params.gpus_per_node=1 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/new_circle_3 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/new_circle_3 \
  data.train_dataloader_params.pushshapes_sim.batch_size=256 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=256 \
  data.train_dataloader_params.pushshapes_sim.num_workers=8 \
  data.valid_dataloader_params.pushshapes_sim.num_workers=8 \
  trainer.max_epochs=500 \
  trainer.precision=32 \
  trainer.check_val_every_n_epoch=100 \
  trainer.limit_val_batches=10 \
    norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/datasets/new_circle_3/norm_stats.json \
  model.optimizer.lr=1e-4
