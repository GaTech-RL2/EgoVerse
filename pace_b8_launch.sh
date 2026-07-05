#!/bin/bash
# baseline8 bottom-allocation sweep on 1xH200 (global batch 33 = skynet 3x11 parity).
G=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
DS=/storage/project/r-dxu345-0/paphiwetsa3/datasets
cd $G
export PYTHONPATH=. MUJOCO_GL=egl
export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"
PY=$G/.venv/bin/python
cell=$1
CKPT=$(ls -t logs/indomain_c4_200m_v5/${cell}_2026*/checkpoints/last.ckpt logs/indomain_c4_200m_v5/${cell}_2026*/csv_logs/lightning_logs/version_0/checkpoints/last.ckpt 2>/dev/null | head -1)
RES=""
if [ -n "$CKPT" ]; then RES="ckpt_path=$CKPT"; echo "[RESUME] $cell <- $CKPT"; else echo "[FRESH] $cell"; fi
DATA_OV="++data.train_datasets.pushshapes_sim.resolver.folder_path=$DS/circle_3000_plus_gen ++data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS/circle_3000_plus_gen ++data.train_datasets.pushshapes_sim_small_circle.resolver.folder_path=$DS/small_circle_3000 ++data.valid_datasets.pushshapes_sim_small_circle.resolver.folder_path=$DS/small_circle_3000"
DL="++data.train_dataloader_params.pushshapes_sim.batch_size=66 ++data.train_dataloader_params.pushshapes_sim.num_workers=4 ++data.train_dataloader_params.pushshapes_sim_small_circle.batch_size=66 ++data.train_dataloader_params.pushshapes_sim_small_circle.num_workers=4"
srun $PY -m egomimic.trainHydra --config-name=train_zarr_cartesian +experiment=indomain_c4/$cell $RES logger=csv callbacks=ckpt_ratio_loss_scheduler_700 \
  ++callbacks.model_checkpoint.every_n_epochs=250 ++callbacks.model_checkpoint.save_on_train_epoch_end=True \
  +callbacks.ckpt_rolling._target_=lightning.pytorch.callbacks.ModelCheckpoint \
  +callbacks.ckpt_rolling.every_n_epochs=25 +callbacks.ckpt_rolling.save_top_k=1 \
  "+callbacks.ckpt_rolling.monitor=epoch" +callbacks.ckpt_rolling.mode=max \
  +callbacks.ckpt_rolling.save_last=true "+callbacks.ckpt_rolling.filename='rolling-{epoch}'" \
  +callbacks.ckpt_rolling.save_on_train_epoch_end=true \
  "+callbacks.ckpt_rolling.dirpath=checkpoints" \
  ++trainer.precision=32 ++trainer.limit_train_batches=1.0 ++trainer.limit_val_batches=0 ++trainer.num_sanity_val_steps=0 \
  ++trainer.max_epochs=2500 ++trainer.min_epochs=2500 ++trainer.check_val_every_n_epoch=250 \
  ++model.scheduler_interval=epoch ++model.scheduler.max_steps=2500 ++model.scheduler.warmup_steps=125 \
  ++model.log_per_layer_grad_norms=true ++model.enable_grad_norm=false \
  ++callbacks.ratio_loss_scheduler.on_value=1.0 ++callbacks.ratio_loss_scheduler.off_value=1.0 \
  $( [ "$cell" = "hnet_b4c_cotrain" ] && echo "+callbacks.ratio_ctrl._target_=egomimic.pl_utils.callbacks.ratio_loss_controller.RatioLossController +callbacks.ratio_ctrl.tol=0.03 +callbacks.ratio_ctrl.rho=1.2 +callbacks.ratio_ctrl.decay=0.995 +callbacks.ratio_ctrl.every_n_epochs=10 +callbacks.ratio_ctrl.w_min=1.0 +callbacks.ratio_ctrl.w_max=50.0" ) \
  ++norm_stats.norm_mode=minmax ++norm_stats.precomputed_norm_path=$G/b8_norm_stats.json \
  $DATA_OV $DL launch_params.gpus_per_node=1 ++model.optimizer.lr=1.4e-4 ++trainer.strategy=ddp_find_unused_parameters_true \
  name=indomain_c4_200m_v5 description=$cell
