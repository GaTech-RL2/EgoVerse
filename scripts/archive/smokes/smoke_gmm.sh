#!/bin/bash
# GMM head end-to-end training smoke on the fused model.
# Usage: smoke_gmm.sh <action_head> <chunk_k> <desc>
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json

HEAD=${1:-gmm}
CHUNKK=${2:-1}
DESC=${3:-smk_gmm}

# action_head override only added when not "continuous" so the default path is
# byte-identical (we ALSO run a continuous baseline below for the regression guard).
EXTRA=""
if [ "$HEAD" != "continuous" ]; then
  EXTRA="+model.robomimic_model.action_head=${HEAD} +model.robomimic_model.gmm_num_modes=5"
fi

python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_gmm_smoke description=${DESC} mode=train data=tsimulation \
  model=hnet_pushshapes_fused_pusher \
  +model.robomimic_model.token_dropout_p=1.0 \
  +model.robomimic_model.chunk_k=${CHUNKK} \
  ${EXTRA} \
  ++model.robomimic_model.action_horizon=1024 \
  ++model.robomimic_model.cond_encoder.img_encoders.front_img_1.spatial=true \
  evaluator=eval_hnet_sim \
  evaluator.init_seeds='[0,1]' evaluator.max_steps=20 \
  evaluator.coverage_threshold=0.8 evaluator.limit_val_batches=1 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=100 \
  trainer=debug trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=6 trainer.limit_val_batches=1 \
  trainer.check_val_every_n_epoch=2 \
  trainer.profiler=null logger=csv \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.precomputed_norm_path=$NORM
echo "TRAIN_EXIT=$?"
