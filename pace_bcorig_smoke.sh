#!/bin/bash
# FAITHFUL replica of the original 0.886 hnet_bc_big (interleaved pyramid).
# Recipe matched to logs/_broken_hnet_pyramid/..._21-25-30: bf16, bs16 (=orig
# 2x8 global), limit_train_batches=100, max_epochs 3000, AdamW lr1e-4 +
# warmup_cosine 300000/15000 (baked into the model yaml -- NOT overridden here).
G=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
DS=/storage/project/r-dxu345-0/paphiwetsa3/datasets
cd $G
export PYTHONPATH=. MUJOCO_GL=egl
export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"
PY=$G/.venv/bin/python
cell=hnet_bc_big_orig
CKPT=$(ls -t logs/indomain_c4_200m_v5/${cell}_2026*/checkpoints/last.ckpt logs/indomain_c4_200m_v5/${cell}_2026*/csv_logs/lightning_logs/version_0/checkpoints/last.ckpt 2>/dev/null | head -1)
RES=""; if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then RES="ckpt_path=$CKPT"; echo "[RESUME] $cell <- $CKPT"; else echo "[FRESH] $cell"; fi
DATA_OV="++data.train_datasets.pushshapes_sim.resolver.folder_path=$DS/circle_3000_plus_gen ++data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS/circle_3000_plus_gen"
DL="++data.train_dataloader_params.pushshapes_sim.batch_size=16 ++data.train_dataloader_params.pushshapes_sim.num_workers=8 ++data.train_dataloader_params.pushshapes_sim.prefetch_factor=4 ++data.train_dataloader_params.pushshapes_sim.persistent_workers=true"
srun $PY -m egomimic.trainHydra --config-name=train_zarr_cartesian +experiment=indomain_c4/$cell $RES logger=csv callbacks=checkpoints \
  ++callbacks.model_checkpoint.every_n_epochs=200 ++callbacks.model_checkpoint.save_on_train_epoch_end=True ++callbacks.model_checkpoint.save_top_k=-1 \
  +callbacks.ckpt_rolling._target_=lightning.pytorch.callbacks.ModelCheckpoint \
  +callbacks.ckpt_rolling.every_n_epochs=25 +callbacks.ckpt_rolling.save_top_k=1 \
  "+callbacks.ckpt_rolling.monitor=epoch" +callbacks.ckpt_rolling.mode=max \
  +callbacks.ckpt_rolling.save_last=true "+callbacks.ckpt_rolling.filename='rolling-{epoch}'" \
  +callbacks.ckpt_rolling.save_on_train_epoch_end=true \
  "+callbacks.ckpt_rolling.dirpath=checkpoints" \
  ++trainer.precision=bf16 ++trainer.limit_train_batches=2 ++trainer.limit_val_batches=0 ++trainer.num_sanity_val_steps=0 \
  ++trainer.max_epochs=2 ++trainer.min_epochs=2 ++trainer.check_val_every_n_epoch=1000 ++norm_stats.sample_frac=0.1 \
  ++model.log_per_layer_grad_norms=true ++model.enable_grad_norm=false \
  ++norm_stats.norm_mode=minmax ++norm_stats.precomputed_norm_path=$G/b8_norm_stats.json \
  $DATA_OV $DL launch_params.gpus_per_node=1 ++trainer.strategy=ddp_find_unused_parameters_true \
  name=debug description=bcorig_smoke
