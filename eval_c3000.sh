#!/bin/bash
#SBATCH -J hptFlowC3000Causal_eval
#SBATCH --gres=gpu:a40:1
#SBATCH -p rl2-lab
#SBATCH -A rl2-lab
#SBATCH -c 12
#SBATCH --mem=30G

cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source .venv/bin/activate

/coc/flash7/paphiwetsa3/projects/EgoVerse2/.venv/bin/python -m egomimic.trainHydra \
  name=hptFlowC3000Causal_eval \
  description=eval_circle_3000_causal \
  mode=eval \
  ckpt_path=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC3000Causal/hpt_flow_circle_3000_causal_obs1_act32_2026-06-13_03-08-51/checkpoints/last.ckpt \
  data=tsimulation_hpt \
  model=hpt_pushshapes_circle \
  evaluator=eval_hpt_standard \
  +evaluator.rollout_mode=chunk_openloop \
  +evaluator.chunk_k=32 \
  +evaluator.temporal_ensemble=false \
  +evaluator.limit_val_batches=20 \
  trainer.limit_val_batches=20 \
  logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/circle_3000 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/circle_3000 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal
