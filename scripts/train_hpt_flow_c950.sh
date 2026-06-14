#!/bin/bash
#SBATCH --job-name=hptFlowC950Causal
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hptflow_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hptflow_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export WANDB_ENTITY=rl2-group WANDB_MODE=online

NAME=hptFlowC950Causal
# Same config as hptFlowC3000Causal (run 3334349) — ONLY the dataset changes to the OLD ~950
# circle data (new_circle_3, 954 eps) for a data-quantity comparison (950 vs 3000).
# Fresh norm (computed on new_circle_3, quantile default — different data than circle_3000).
DS=/coc/flash7/paphiwetsa3/datasets/new_circle_3

# requeue-safe: auto-resume from this run's newest last.ckpt on a scavenger preemption
RESUME=""
LAST=$(ls -t /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/${NAME}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
[ -n "$LAST" ] && RESUME="ckpt_path=$LAST"

srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=hpt_flow_new_circle_3_causal_obs1_act32 mode=train \
  data=tsimulation_hpt model=hpt_pushshapes_circle \
  model.optimizer.lr=4e-5 '~model.scheduler' \
  +model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  +model.scheduler._partial_=true +model.scheduler.max_steps=90000 \
  +model.scheduler.warmup_steps=500 +model.scheduler.warmup_start_factor=0.1 +model.scheduler.eta_min=4.0e-6 \
  evaluator=eval_hpt_standard \
  +evaluator.rollout_mode=chunk_openloop +evaluator.chunk_k=32 +evaluator.temporal_ensemble=false \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=100 \
  trainer=debug trainer.max_epochs=1800 trainer.limit_train_batches=50 \
  trainer.check_val_every_n_epoch=100 trainer.limit_val_batches=4 \
  trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
  ${RESUME}
echo "TRAIN_EXIT=$?"
