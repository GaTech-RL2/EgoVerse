#!/bin/bash
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=offline
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_baseline_nc3 description=smoke_chunkA mode=train data=tsimulation \
  model=hnet_pushshapes_fused_pusher_resnet \
  +model.robomimic_model.token_dropout_p=1.0 \
  ++model.robomimic_model.chunk_k=32 \
  ++model.robomimic_model.action_horizon=1024 \
  ++model.robomimic_model.cond_encoder.img_encoders.front_img_1.spatial=true \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=chunk_openloop evaluator.chunk_k=32 \
  trainer=debug trainer.devices=1 trainer.max_epochs=4 trainer.min_epochs=4 \
  trainer.limit_train_batches=4 trainer.limit_val_batches=1 trainer.check_val_every_n_epoch=4 \
  trainer.profiler=null logger=wandb \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=4 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=64 evaluator.coverage_threshold=0.8 \
  evaluator.init_seeds="[0,1]" evaluator.limit_val_batches=1 \
  norm_stats.precomputed_norm_path=$NORM
echo "SMOKE_CHUNKA_EXIT=$?"
