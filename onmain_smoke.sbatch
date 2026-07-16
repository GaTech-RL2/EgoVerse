#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --exclude=ig-88,spd-13,sonny,cyborg,hk47,heistotron,baymax,deebot,kitt,cheetah,megazord,puma,omgwth
#SBATCH --output=onmain_smoke_%j.out
set -uo pipefail
W=/coc/flash7/paphiwetsa3/projects/bf_onmain_wt
cd $W
export PYTHONPATH=$W MUJOCO_GL=egl HYDRA_FULL_ERROR=1
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow/.venv/bin/python
$PY -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  +experiment=indomain_c4/dualstream_cotrain_2trunk_d16_200m \
  model=bf_prdec_nopre \
  description=onmain_smoke trainer=ddp_pi \
  launch_params.gpus_per_node=1 launch_params.nodes=1 ++trainer.strategy=auto \
  model.enable_grad_norm=false \
  norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/datasets/pushshapes_norm_stats_ws512.json \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/probe_1ep_circle \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/probe_1ep_circle \
  data.train_datasets.pushshapes_sim_small_circle.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/probe_1ep_smallcircle \
  data.valid_datasets.pushshapes_sim_small_circle.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/probe_1ep_smallcircle \
  trainer.max_epochs=40 trainer.min_epochs=40 ++trainer.limit_train_batches=8 \
  trainer.limit_val_batches=1 trainer.check_val_every_n_epoch=40 \
  +model.scheduler_interval=epoch model.scheduler.max_steps=40 model.scheduler.warmup_steps=2 \
  callbacks.model_checkpoint.save_last=true callbacks.model_checkpoint.every_n_epochs=40 \
  logger=csv
echo ONMAIN_SMOKE_rc=$?
