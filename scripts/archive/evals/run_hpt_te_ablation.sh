#!/bin/bash
#SBATCH --job-name=hpt-te-abl
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hpt_te_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hpt_te_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=.
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
CKPT=logs/hpt_pusher_nc3/reg_causal_pusher_500ep_2026-05-31_01-34-41/checkpoints/hpt_pusher_ep499.ckpt
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=logs/hpt_pusher_nc3/reg_causal_pusher_500ep_2026-05-31_01-34-41/norm_stats/norm_stats.json
TE=${TE_MODE:-true}
srun python scripts/fast_dataloader_wrapper.py \
  --config-name=train_zarr_cartesian \
  name=hpt_te_ablation description=hpt_te_${TE}_20seed \
  mode=eval \
  model=hpt_pushshapes_circle_regression \
  data=tsimulation \
  +data_schematic=hpt \
  evaluator=eval_sim_only \
  evaluator.temporal_ensemble=${TE} \
  ckpt_path=$CKPT \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_dataloader_params.pushshapes_sim.batch_size=20 \
  trainer.devices=1 \
  trainer.limit_val_batches=1 \
  evaluator.limit_val_batches=1 \
  norm_stats.precomputed_norm_path=$NORM
echo "EXIT_CODE=$?"
