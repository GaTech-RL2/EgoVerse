#!/bin/bash
# hptFlowComboEma smoke: 2 epochs x 2 batches, csv logger, wandb off.
# Runs python DIRECTLY (invoked under srun already; no nested srun).
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
DS=/coc/flash7/paphiwetsa3/datasets/circle_3000
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/external_ckpts/hptFlowC3000_circle3000_norm_stats.json
python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hptFlowComboEma_smoke description=hptFlowComboEma_smoke mode=train \
  data=tsimulation_hpt model=hpt_bc_flow_pushshapes_ema \
  model.optimizer.lr=4e-5 \
  '~model.scheduler' \
  +model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  +model.scheduler._partial_=true \
  +model.scheduler.max_steps=12 \
  +model.scheduler.warmup_steps=2 \
  +model.scheduler.warmup_start_factor=0.1 \
  +model.scheduler.eta_min=4.0e-6 \
  evaluator=eval_hpt_standard \
  callbacks=checkpoints_ema callbacks.model_checkpoint.every_n_epochs=2 \
  trainer=debug trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=2 \
  trainer.profiler=null logger=csv \
  norm_stats.precomputed_norm_path=$NORM \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS
echo "SMOKE_RC=$?"
