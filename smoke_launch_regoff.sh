#!/bin/bash
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
echo "== SMOKE: bf_prdec_regoff_ste4_499 (init_ckpt + ratio 0 + ste_gain 4) =="
srun -A hoffman-lab -p hoffman-lab --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G -t 00:45:00 \
  --exclude=omgwth,cyborg,ig-88,hk47,spd-13,sonny,kitt,cheetah,heistotron,megazord,puma,baymax,deebot \
  .venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  +experiment=indomain_c4/dualstream_cotrain_2trunk_d16_200m \
  model=bf_prdec_regoff_ste4_499 trainer=ddp_pi description=smoke_regoff_ste4 \
  launch_params.gpus_per_node=1 launch_params.nodes=1 ++trainer.strategy=auto \
  model.enable_grad_norm=false \
  norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/datasets/pushshapes_norm_stats_ws512.json \
  trainer.max_epochs=2 trainer.min_epochs=2 ++trainer.limit_train_batches=2 \
  trainer.limit_val_batches=0 trainer.check_val_every_n_epoch=2 \
  +model.scheduler_interval=epoch model.scheduler.max_steps=3000 model.scheduler.warmup_steps=150
RC=$?
echo "SMOKE_RC=$RC"
[ $RC -ne 0 ] && { echo "SMOKE FAILED - NOT LAUNCHING"; exit 1; }
for M in bf_prdec_regoff499 bf_prdec_regoff999 bf_prdec_ste4_499 bf_prdec_regoff_ste4_499; do
  J=$(MODEL=$M NAME=$M sbatch --parsable -J "$M" bf_prdec_abl.sbatch)
  echo "LAUNCHED $M job=$J"
done
echo ALL_LAUNCHED
