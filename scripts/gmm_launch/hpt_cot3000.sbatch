#!/bin/bash
#SBATCH --job-name=hpt_cot3000
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250G
#SBATCH --time=3-00:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/hpt_cot3000_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
srun .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hpt_cot3000 description=hpt_cot3000_200k \
  mode=train \
  data=tsimulation_hpt_cot3000 model=hpt_cotrain_pushshapes_flow_shared_head evaluator=eval_hpt_standard trainer=ddp_pi \
  model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  ~model.scheduler.start_factor ~model.scheduler.end_factor \
  +model.scheduler.max_steps=200000 +model.scheduler.warmup_steps=2000 \
  +model.scheduler.warmup_start_factor=0.1 +model.scheduler.eta_min=1e-6 \
  norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=null trainer.precision=32 data.train_datasets.pushshapes_sim.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/circle_3000_plus_gen data.valid_datasets.pushshapes_sim.resolver.folder_path=/storage/project/r-dxu345-0/paphiwetsa3/datasets/circle_3000_plus_gen \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=500 \
  trainer.max_epochs=2000 trainer.min_epochs=1 \
  trainer.limit_train_batches=100 trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=500 \
  logger=csv
