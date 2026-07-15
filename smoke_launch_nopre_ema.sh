#!/bin/bash
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
echo "== SMOKE: bf_nopre_ema_epcrop (EMA + GN + episode-scope crop) =="
srun -A hoffman-lab -p hoffman-lab --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G -t 00:45:00 \
  --exclude=omgwth,cyborg,ig-88,hk47,spd-13,sonny,kitt,cheetah,heistotron,megazord,puma,baymax,deebot,megabot \
  .venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  +experiment=indomain_c4/dualstream_cotrain_2trunk_d16_200m \
  model=bf_nopre_ema_epcrop trainer=ddp_pi description=smoke_epcrop \
  callbacks=checkpoints_ema \
  launch_params.gpus_per_node=1 launch_params.nodes=1 ++trainer.strategy=auto \
  model.enable_grad_norm=false \
  norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/datasets/pushshapes_norm_stats_ws512.json \
  trainer.max_epochs=2 trainer.min_epochs=2 ++trainer.limit_train_batches=2 \
  trainer.limit_val_batches=0 trainer.check_val_every_n_epoch=2 \
  +model.scheduler_interval=epoch model.scheduler.max_steps=3000 model.scheduler.warmup_steps=150
RC=$?
echo "SMOKE_RC=$RC"
[ $RC -ne 0 ] && { echo "SMOKE FAILED - NOT LAUNCHING"; exit 1; }
for SPEC in "bf_nopre_ema" "bf_nopre_ema_epcrop"; do
  J=$(MODEL=$SPEC NAME=$SPEC EXTRA="callbacks=checkpoints_ema" sbatch --parsable -J "$SPEC" bf_prdec_abl.sbatch)
  echo "LAUNCHED $SPEC job=$J"
done
echo ALL_LAUNCHED
