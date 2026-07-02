#!/bin/bash
# 200M H-Net + Tx launcher (BASELINE v5). Relaunch of v2/v4 with the obs_stride bug fixed
# in code (egomimic/algo/hnet/algo.py now defaults obs_stride to the GMM head chunk_len=4,
# was silently 1). FRESH runs (cannot resume obs_stride=1 ckpts) under NEW name
# indomain_c4_200m_v5 + wandb group indomain_c4_baseline8. Changes vs v2/v4:
#   - 3-GPU DDP (gpus_per_node=3, ntasks-per-node=3)
#   - per-GPU batch 11 -> global 33 (~2x the old 8/GPU x 2 = 16) via per-cell dataloader keys
#   - limit_train_batches=1.0 (full pass, no 100-batch cap)
#   - max_epochs=min_epochs=2500, ckpt+val every 25 (= 100/4)
#   - num_workers=12 (was 8) on the dataloaders
#   - scheduler max_steps/warmup_steps = <FILL FROM SMOKE: 100*B / round(0.05*100*B)>
# obs_stride must resolve to 4 (verify in log). RatioLossScheduler (switch_fraction=0.35)
# auto-scales off-epoch to round(0.35*max_epochs)=35 with max_epochs=2500 (no rescale needed).
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export TORCH_NCCL_ENABLE_MONITORING=0
export WANDB_INIT_TIMEOUT=300
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/python
cell=$1

# SMOKE branch: when SMOKE=1, append fast overrides (last-wins in Hydra) to verify a
# ckpt saves WITHOUT val before the real relaunch. Reuses ALL v5 overrides untouched.
SMOKE_OV=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_OV="++trainer.limit_train_batches=5 ++trainer.max_epochs=4 ++trainer.min_epochs=4 ++callbacks.model_checkpoint.every_n_epochs=2 ++trainer.limit_val_batches=0 ++norm_stats.sample_frac=0.1 ++model.scheduler.max_steps=4 ++model.scheduler.warmup_steps=1 name=v5_ckpttest description=${cell}_smoke"
fi

# FRESH v5: resume only from v5 logs (so it never picks up a baseline4 obs_stride=1 ckpt).
CKPT=$(ls -t logs/indomain_c4_200m_v5/${cell}_2026*/checkpoints/last.ckpt 2>/dev/null | head -1)
RES=""
if [ -n "$CKPT" ]; then RES="ckpt_path=$CKPT"; echo "[RESUME] $cell <- $CKPT"; else echo "[FRESH] $cell"; fi

case $cell in tx_*) export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;; esac

# Per-cell dataloader batch_size + num_workers overrides. bc_big -> pushshapes_sim only;
# bc_small -> pushshapes_sim_small_circle only; cotrain/dino -> both (a phantom key on a
# single-dataset cell is a silent no-op, but we key it correctly to avoid surprises).
BS=11; NW=12
DL=""
case $cell in
  *bc_big*|tx_bc_big)
    DL="++data.train_dataloader_params.pushshapes_sim.batch_size=$BS ++data.train_dataloader_params.pushshapes_sim.num_workers=$NW";;
  *bc_small*)
    DL="++data.train_dataloader_params.pushshapes_sim_small_circle.batch_size=$BS ++data.train_dataloader_params.pushshapes_sim_small_circle.num_workers=$NW";;
  *cotrain*)
    DL="++data.train_dataloader_params.pushshapes_sim.batch_size=$BS ++data.train_dataloader_params.pushshapes_sim.num_workers=$NW ++data.train_dataloader_params.pushshapes_sim_small_circle.batch_size=$BS ++data.train_dataloader_params.pushshapes_sim_small_circle.num_workers=$NW";;
esac

srun $PY -m egomimic.trainHydra --config-name=train_zarr_cartesian +experiment=indomain_c4/$cell $RES logger=csv_wandb callbacks=ckpt_ratio_loss_scheduler_700 ++callbacks.model_checkpoint.every_n_epochs=250 ++callbacks.model_checkpoint.save_on_train_epoch_end=True ++trainer.precision=32 ++trainer.limit_train_batches=1.0 ++trainer.limit_val_batches=0 ++trainer.num_sanity_val_steps=0 ++trainer.max_epochs=2500 ++trainer.min_epochs=2500 ++trainer.check_val_every_n_epoch=250 ++model.scheduler_interval=epoch ++model.scheduler.max_steps=2500 ++model.scheduler.warmup_steps=125 ++model.log_per_layer_grad_norms=true ++model.enable_grad_norm=false ++callbacks.ratio_loss_scheduler.on_value=1.0 ++callbacks.ratio_loss_scheduler.off_value=1.0 ++norm_stats.norm_mode=minmax $DL launch_params.gpus_per_node=3 ++trainer.strategy=ddp_find_unused_parameters_true ++logger.wandb.project=pushshapes-indomain-c4 ++logger.wandb.id=indomain_c4_baseline8_$cell ++logger.wandb.name=${cell}_200m_ed ++logger.wandb.group=indomain_c4_baseline8 ++logger.wandb.resume=allow name=indomain_c4_200m_v5 description=$cell $SMOKE_OV
