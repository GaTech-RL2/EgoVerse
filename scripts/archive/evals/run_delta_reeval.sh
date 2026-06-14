#!/bin/bash
# Re-eval the trained delta checkpoints with the FIXED rollout seed (action[0]
# instead of agent_pos). No retrain — just rerun the AR rollout. Usage:
#   bash run_delta_reeval.sh <run_dir> <model> <desc>
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl WANDB_MODE=disabled HYDRA_FULL_ERROR=1

export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
RD="$1"; MODEL="$2"; DESC="$3"
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=delta_reeval description=$DESC \
  mode=eval \
  data=tsimulation_delta \
  model=$MODEL \
  evaluator=eval_hnet_sim \
  trainer=debug \
  trainer.devices=1 \
  trainer.limit_val_batches=4 \
  trainer.profiler=null \
  logger=csv \
  ckpt_path=$RD/checkpoints/ep79.ckpt \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.delta_action=true \
  evaluator.max_steps=null \
  norm_stats.precomputed_norm_path=$RD/norm_stats/norm_stats.json 2>&1
echo "EXIT_CODE=$?"
