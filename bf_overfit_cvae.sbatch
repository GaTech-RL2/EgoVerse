#!/bin/bash
#SBATCH --job-name=bf_overfit_cvae
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --account=hoffman-lab
#SBATCH --partition=hoffman-lab
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --time=6:00:00
#SBATCH --exclude=kitt,cyborg,ig-88,hk47,spd-13,sonny,heistotron,megazord,puma,baymax,deebot,cheetah
#SBATCH --output=bf_overfit_cvae_%j.out
# BATCHFLOW EQUIVALENCE GATE part 1: sym_w1 (GMM) 1-ep overfit on the NEW code.
# Old-code reference (frozen repo): NLL descends steeply on 1 ep; TF preds ~= GT.
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
srun .venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  +experiment=indomain_c4/dualstream_cotrain_m16 \
  model=bf_cvae \
  trainer=ddp_pi description=bf_overfit_cvae name=debug \
  launch_params.gpus_per_node=1 launch_params.nodes=1 ++trainer.strategy=auto \
  model.enable_grad_norm=false \
  logger=csv \
  norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/datasets/pushshapes_norm_stats_ws512.json \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/probe_1ep_circle \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/probe_1ep_circle \
  data.train_datasets.pushshapes_sim_small_circle.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/probe_1ep_smallcircle \
  data.valid_datasets.pushshapes_sim_small_circle.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/probe_1ep_smallcircle \
  data.train_dataloader_params.pushshapes_sim.num_workers=2 \
  data.train_dataloader_params.pushshapes_sim_small_circle.num_workers=2 \
  data.valid_dataloader_params.pushshapes_sim.num_workers=2 \
  data.valid_dataloader_params.pushshapes_sim_small_circle.num_workers=2 \
  data.train_dataloader_params.pushshapes_sim.batch_size=4 \
  data.train_dataloader_params.pushshapes_sim_small_circle.batch_size=4 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=4 \
  data.valid_dataloader_params.pushshapes_sim_small_circle.batch_size=4 \
  trainer.max_epochs=1000 trainer.min_epochs=1000 \
  trainer.check_val_every_n_epoch=500 \
  callbacks.model_checkpoint.save_last=true callbacks.model_checkpoint.save_top_k=1 \
  callbacks.model_checkpoint.every_n_epochs=500 \
  model.scheduler.max_steps=1000 model.scheduler.warmup_steps=50
