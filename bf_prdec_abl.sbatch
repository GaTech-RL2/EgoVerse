#!/bin/bash
#SBATCH --job-name=bf_abl
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --account=hoffman-lab
#SBATCH --partition=hoffman-lab
#SBATCH --qos=long
#SBATCH --time=6-12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=omgwth,cyborg,ig-88,hk47,spd-13,sonny,kitt,cheetah,heistotron,megazord,puma,baymax,deebot
#SBATCH --output=bf_abl_%x_%j.out
# CVAE_ZS (v2, PRIORITY ARM) — full pushT cotrain on the BATCHFLOW code.
# z-in-S via AdaLN (adaLN-zero, A untouched), raw-S pre-trunk posterior,
# z->0 sampling 0.1, band sg=on, fb=0.5. bcrecipe bs16/ltb100/3000ep.
# Ladder: GMM bcrecipe (0.809/0.448@999) -> CVAE v1 (running) -> THIS.
MODEL=${MODEL:?}; NAME=${NAME:?}; EXTRA=${EXTRA:-}
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
CELL=$NAME
LAST=$(ls -t logs/indomain_c4/${CELL}_2026*/checkpoints/last.ckpt 2>/dev/null | head -1)
RES=""; [ -n "$LAST" ] && RES="ckpt_path=$LAST"; echo "[${LAST:+RESUME}${LAST:-FRESH}] $CELL"
srun .venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian $RES \
  +experiment=indomain_c4/dualstream_cotrain_2trunk_d16_200m \
  model=$MODEL \
  trainer=ddp_pi description=$CELL \
  launch_params.gpus_per_node=1 launch_params.nodes=1 ++trainer.strategy=auto \
  model.enable_grad_norm=false \
  norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/datasets/pushshapes_norm_stats_ws512.json \
  trainer.max_epochs=3000 trainer.min_epochs=3000 \
  ++trainer.limit_train_batches=100 trainer.limit_val_batches=100000 \
  trainer.check_val_every_n_epoch=250 \
  +model.scheduler_interval=epoch model.scheduler.max_steps=3000 model.scheduler.warmup_steps=150 \
  callbacks.model_checkpoint.save_last=true callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_epochs=500 ++callbacks.model_checkpoint.save_on_train_epoch_end=true \
  $EXTRA
