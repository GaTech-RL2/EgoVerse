#!/bin/bash
# Args: MODEL CKPT RUNNAME
set -uo pipefail
MODEL="$1"; CKPT="$2"; RUNNAME="$3"
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NCS=/coc/flash7/paphiwetsa3/datasets/new_circle_small__3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetSmallC8FHR_nc3/bc_rnn_hnet_smallcircle_c8_fhr_2026-06-07_16-11-07/norm_stats/norm_stats.json
python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${RUNNAME} description=${RUNNAME} mode=eval data=tsimulation \
  model=${MODEL} \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar evaluator.init_mode=seeds \
  evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  ++evaluator.env_kwargs.pusher_shape=circle_small \
  trainer=debug trainer.precision=32 trainer.limit_val_batches=1 \
  logger=csv \
  "ckpt_path='${CKPT}'" \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NCS \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  '~data.valid_datasets.pushshapes_sim' \
  '+data.valid_datasets.pushshapes_sim._target_=egomimic.rldb.zarr.zarr_dataset_packed.ZarrEpisodePackedDataset.from_resolver' \
  '+data.valid_datasets.pushshapes_sim.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver' \
  +data.valid_datasets.pushshapes_sim.resolver.folder_path=$NCS \
  +data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  +data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=1024 \
  '+data.valid_datasets.pushshapes_sim.resolver.transform_list=null' \
  '+data.valid_datasets.pushshapes_sim.chunking=none' \
  +data.valid_datasets.pushshapes_sim.min_seq_len=64 \
  '+data.valid_datasets.pushshapes_sim.max_seq_len=null' \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=$NORM
echo "EVAL_EXIT=$?"
