#!/bin/bash
#SBATCH -A rl2-lab
#SBATCH --partition=rl2-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --job-name=hpt_reg_zero_pad
#SBATCH -o /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_data_ablation/hpt_reg_zero_padded_%j.out
#SBATCH -e /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_data_ablation/hpt_reg_zero_padded_%j.err
set -euxo pipefail
mkdir -p /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_data_ablation
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=.
export MUJOCO_GL=egl
export WANDB_ENTITY=rl2-group

python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hpt_data_ablation \
  description=reg_zero_padded_491ep_2xA40 \
  mode=train \
  model=hpt_pushshapes_circle_regression \
  data=tsimulation_hpt \
  +data_schematic=hpt \
  evaluator=eval_hpt_standard \
  launch_params.gpus_per_node=2 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_zero_padded \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_zero_padded \
  data.train_dataloader_params.pushshapes_sim.batch_size=512 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=512 \
  data.train_dataloader_params.pushshapes_sim.num_workers=12 \
  data.valid_dataloader_params.pushshapes_sim.num_workers=12 \
  +data.train_dataloader_params.pushshapes_sim.pin_memory=true \
  +data.train_dataloader_params.pushshapes_sim.persistent_workers=true \
  +data.train_dataloader_params.pushshapes_sim.prefetch_factor=4 \
  trainer.max_epochs=500 \
  trainer.precision=bf16-mixed \
  trainer.limit_val_batches=2 \
  model.optimizer.lr=4e-4
