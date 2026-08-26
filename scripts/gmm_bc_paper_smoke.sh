#!/bin/bash
# Quick 2-epoch GMM-BC smoke on pushshapes_paper (fixed-target boundary task).
# Verifies folder_path=pushshapes_paper + get_keymap_eval + minmax norm LOADS and trains.
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export WANDB_MODE=disabled
PAPER=/coc/flash7/paphiwetsa3/datasets/pushshapes_paper
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=gmmBCpaperSmoke description=gmm_bc_paper_smoke mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_tx_chunk8 \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
  evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=2 \
  trainer=debug trainer.precision=32 trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=4 trainer.limit_val_batches=2 trainer.check_val_every_n_epoch=2 \
  trainer.profiler=null logger=csv \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$PAPER \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$PAPER \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax
echo "SMOKE_EXIT=$?"
