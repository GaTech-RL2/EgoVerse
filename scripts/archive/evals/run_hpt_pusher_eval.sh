#!/bin/bash
# HPT reg-499 sim-rollout smoke on new_circle_3 (packed eval data + get_keymap_eval goal).
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

python scripts/fast_dataloader_wrapper.py \
  --config-name=train_zarr_cartesian \
  name=hpt_eval_smoke description=hpt_pusher_ep499 \
  mode=eval \
  model=hpt_pushshapes_circle_regression \
  data=tsimulation \
  +data_schematic=hpt \
  evaluator=eval_sim_only \
  ckpt_path=$CKPT \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_dataloader_params.pushshapes_sim.batch_size=8 \
  trainer.devices=1 \
  trainer.limit_val_batches=1 \
  evaluator.limit_val_batches=1 \
  norm_stats.precomputed_norm_path=logs/hpt_pusher_nc3/reg_causal_pusher_500ep_2026-05-31_01-34-41/norm_stats/norm_stats.json
echo "EXIT_CODE=$?"
