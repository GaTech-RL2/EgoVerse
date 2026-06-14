#!/bin/bash
# HPT regression baseline, PUSHER-ONLY state, on new_circle_3. Mirrors the
# reg_causal_4xa40 recipe that produced the 0.54 (5-dim) baseline; the config is
# now input_slice [0,2] so the model sees only agent xy. Trains only (causal data
# isn't packed); sim eval is a separate mode=eval pass afterward.
#SBATCH --job-name=hpt-reg-pusher
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hpt_pusher_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hpt_pusher_%j.err
set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
mkdir -p logs/sbatch
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
hostname; nvidia-smi || true

export PYTHONPATH=.
export MUJOCO_GL=egl
export WANDB_MODE=online
export WANDB_ENTITY=rl2-group
export HYDRA_FULL_ERROR=1
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_inmem/reg_causal_4xa40_2026-05-30_12-25-07/norm_stats/norm_stats.json

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hpt_pusher_nc3 description=hpt_simpleconv_500ep \
  mode=train \
  model=hpt_pushshapes_simpleconv \
  data=tsimulation_hpt_causal \
  +data_schematic=hpt \
  ~evaluator \
  logger=wandb \
  launch_params.gpus_per_node=4 \
  trainer.devices=4 \
  trainer.strategy=ddp_find_unused_parameters_true \
  trainer.max_epochs=500 \
  trainer.precision=32 \
  trainer.limit_train_batches=25 \
  trainer.check_val_every_n_epoch=100 \
  trainer.limit_val_batches=10 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_dataloader_params.pushshapes_sim.batch_size=128 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=128 \
  data.train_dataloader_params.pushshapes_sim.num_workers=7 \
  data.valid_dataloader_params.pushshapes_sim.num_workers=7 \
  norm_stats.precomputed_norm_path=$NORM \
  model.optimizer.lr=4e-4
echo "EXIT_CODE=$?"
