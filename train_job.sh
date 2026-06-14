#!/bin/bash
#SBATCH -J hptFlowC200Causal_pushshapesPaper
#SBATCH --gres=gpu:a40:1
#SBATCH -p rl2-lab
#SBATCH -A rl2-lab
#SBATCH -c 12
#SBATCH --mem=30G
#SBATCH -o logs/train_%j.log

cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source .venv/bin/activate

/coc/flash7/paphiwetsa3/projects/EgoVerse2/.venv/bin/python -m egomimic.trainHydra \
  name=hptFlowC200Causal_pushshapesPaper \
  description=hpt_flow_pushshapes_paper_causal_obs1_act32 \
  mode=train \
  data=tsimulation_hpt \
  model=hpt_pushshapes_circle \
  model.optimizer.lr=4e-5 \
  ~model.scheduler \
  +model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  +model.scheduler._partial_=true \
  +model.scheduler.max_steps=90000 \
  +model.scheduler.warmup_steps=500 \
  +model.scheduler.warmup_start_factor=0.1 \
  +model.scheduler.eta_min=4.0e-6 \
  evaluator=eval_hpt_standard \
  +evaluator.rollout_mode=chunk_openloop \
  +evaluator.chunk_k=32 \
  +evaluator.temporal_ensemble=false \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=100 \
  trainer=debug \
  trainer.max_epochs=1800 \
  trainer.min_epochs=1800 \
  trainer.limit_train_batches=50 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=100 \
  trainer.profiler=null \
  logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushshapes_paper \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushshapes_paper \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal
