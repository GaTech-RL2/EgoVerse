#!/bin/bash
# Re-eval a trained H-Net checkpoint with chunk+TE rollout (H4 test). Args: <run_dir> <model> <desc>
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl WANDB_MODE=disabled HYDRA_FULL_ERROR=1 PACK_COLLATE_MAX_TOTAL_FRAMES=3200
RD="$1"; MODEL="$2"; DESC="$3"
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=chunkte_reeval description=$DESC mode=eval data=tsimulation model=$MODEL \
  evaluator=eval_hnet_sim trainer=debug trainer.devices=1 trainer.limit_val_batches=4 \
  trainer.profiler=null logger=csv ckpt_path=$RD/checkpoints/ep499.ckpt \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.rollout_mode=chunk_te evaluator.chunk_k=32 evaluator.max_steps=null \
  norm_stats.precomputed_norm_path=$RD/norm_stats/norm_stats.json 2>&1
echo "EXIT_CODE=$?"
