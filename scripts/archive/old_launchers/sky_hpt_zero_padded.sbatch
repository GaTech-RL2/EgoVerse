#!/bin/bash
#SBATCH -A rl2-lab
#SBATCH --partition=rl2-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --job-name=hpt_zero_padded
#SBATCH -o /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_data_ablation/hpt_zero_padded_%j.out
#SBATCH -e /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_data_ablation/hpt_zero_padded_%j.err
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
  description=zero_padded_491ep \
  mode=train \
  model=hpt_pushshapes_circle \
  data=tsimulation_hpt \
  +data_schematic=hpt \
  evaluator=eval_hpt_standard \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_zero_padded \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_zero_padded \
  trainer.max_epochs=400 \
  trainer.precision=bf16-mixed \
  trainer.limit_val_batches=2
