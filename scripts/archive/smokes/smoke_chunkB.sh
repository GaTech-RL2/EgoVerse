#!/bin/bash
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=offline
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
# chunk-B: chunk-rollout closed-loop trainer (obs_source=sim). current_k=#chunks.
MODEL=hnet_pushshapes_fused_windowed_resnet
OBS=${OBS:-sim}
EPOCHS=${EPOCHS:-2}
DESC=${DESC:-smoke_chunkB}
# k_schedule = number of chunks (smoke: 1 chunk = 32 frames every epoch)
KSCHED='[[0,1]]'
srun --jobid=3304844 --overlap --gres=gpu:a40:1 python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=hnet_baseline_nc3 description=${DESC} mode=train data=tsimulation \
  model=${MODEL} \
  ++model.robomimic_model.obs_source=${OBS} \
  ++model.robomimic_model.k_schedule="${KSCHED}" \
  ++model.robomimic_model.max_windows=8 \
  ++model.robomimic_model.max_window_steps=0 \
  ++model.robomimic_model.chunk_rollout=true \
  ++model.robomimic_model.chunk_size=32 \
  ++model.robomimic_model.chunk_k=32 \
  ++model.robomimic_model.n_env_workers=8 \
  +callbacks.kcurriculum._target_=egomimic.algo.hnet_closedloop.KCurriculumCallback \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=chunk_openloop evaluator.chunk_k=32 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=2 \
  trainer=debug trainer.devices=1 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=2 trainer.limit_val_batches=1 trainer.check_val_every_n_epoch=2 \
  trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=64 evaluator.coverage_threshold=0.8 \
  evaluator.init_seeds="[0,1]" evaluator.limit_val_batches=1 \
  norm_stats.precomputed_norm_path=$NORM
echo "SMOKE_CHUNKB_EXIT=$?"
