#!/bin/bash
#SBATCH --job-name=bcrnnHnetC8FHx2
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bcrnn_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bcrnn_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online}
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# ---- CONT-2x of bcrnnHnetC8FH_nc3 (weights-only init + fresh productive warmup-cosine) ----
INIT_CKPT=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetC8FH_nc3/bc_rnn_hnet_chunk8_fullhist_2026-06-10_00-57-36/checkpoints/epoch_epoch=399.ckpt
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetC8FH_nc3/bc_rnn_hnet_chunk8_fullhist_2026-06-10_00-57-36/norm_stats/norm_stats.json
NAME=bcrnnHnetC8FH_nc3_cont2x
DESC=bc_rnn_hnet_chunk8_fullhist_cont2x
EPOCHS=${EPOCHS:-1800}; VALEVERY=100; LTB=50; MAXSTEPS=400
MAXSTEPS_SCHED=$(( EPOCHS * LTB ))   # fresh cosine horizon = new epochs x LTB
EXTRA_ARGS="model.robomimic_model.rnn_horizon=80 model.robomimic_model.core_net.max_window=80"   # per-run hydra overrides (rnn_horizon / window_anchor)

# ---- REQUEUE-SAFE ckpt selection ----
# First launch: weights-only +init_ckpt (installs fresh cosine).
# After a requeue: full ckpt_path resume of THIS run's own newest last.ckpt
# (carries the fresh cosine state), so we never re-init from epoch 0.
LAST=$(ls -t /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/${NAME}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "$LAST" ]; then
  CKPT_MODE="ckpt_path='$LAST'"
  echo "REQUEUE: full-resume from $LAST"
else
  CKPT_MODE="+init_ckpt='$INIT_CKPT'"
  echo "FIRST LAUNCH: weights-only init from $INIT_CKPT"
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_hnet_chunk8 \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=ar \
  evaluator.max_steps=${MAXSTEPS} evaluator.coverage_threshold=0.8 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.precision=32 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=wandb \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
  norm_stats.precomputed_norm_path=$NORM \
  model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  model.scheduler.max_steps=${MAXSTEPS_SCHED} \
  model.scheduler.warmup_steps=4500 \
  model.scheduler.warmup_start_factor=0.1 \
  model.scheduler.eta_min=1.0e-6 \
  $EXTRA_ARGS \
  $CKPT_MODE
echo "TRAIN_EXIT=$?"
