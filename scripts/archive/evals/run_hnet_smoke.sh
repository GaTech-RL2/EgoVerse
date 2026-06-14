#!/bin/bash
# H-Net FUSED (nochunk) 5-epoch smoke on new_circle_3 — validates teacher-forced
# packed training + the new HNetSimEval autoregressive sim-rollout end-to-end.
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=.
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_smoke description=fused_nochunk_nc3_5ep \
  mode=train \
  data=tsimulation \
  model=hnet_pushshapes_fused +model.robomimic_model.token_dropout_p=${HNET_TD:-0.0} ${HNET_EXTRA:-} \
  evaluator=eval_hnet_sim \
  callbacks=checkpoints \
  trainer=debug \
  trainer.max_epochs=5 \
  trainer.min_epochs=5 \
  trainer.limit_train_batches=8 \
  trainer.limit_val_batches=1 \
  trainer.check_val_every_n_epoch=5 \
  trainer.profiler=null \
  logger=csv \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.limit_val_batches=1 \
  evaluator.max_steps=null \
  norm_stats.sample_frac=0.1 2>&1
echo "EXIT_CODE=$?"
