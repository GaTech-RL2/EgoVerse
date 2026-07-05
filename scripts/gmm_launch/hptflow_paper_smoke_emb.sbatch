#!/bin/bash
#SBATCH --job-name=hptflow_paper_smoke
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=embers
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=0:40:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/hptflow_paper_smoke_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
srun .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hptflow_paper_smoke description=hptflow_paper_smoke_200k \
  mode=train \
  data=tsimulation_hpt_paper model=hpt_pushshapes_circle evaluator=eval_hpt_standard trainer=ddp_pi \
  model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  ~model.scheduler.start_factor ~model.scheduler.end_factor \
  +model.scheduler.max_steps=200000 +model.scheduler.warmup_steps=2000 \
  +model.scheduler.warmup_start_factor=0.1 +model.scheduler.eta_min=1e-6 \
  norm_stats.precomputed_norm_path=null \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=2 \
  trainer.max_epochs=2 trainer.min_epochs=1 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=2 \
  logger=csv model.robomimic_model.trunk.action_horizon=8 model.robomimic_model.head_specs.pushshapes_sim.action_horizon=8 model.robomimic_model.head_specs.pushshapes_sim.model.act_seq=8 model.robomimic_model.head_specs.pushshapes_sim.model.hidden_dim=256 data.train_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 norm_stats.norm_mode=minmax trainer.precision=32
