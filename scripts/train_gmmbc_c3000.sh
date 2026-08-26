#!/bin/bash
#SBATCH --job-name=gmmBCc3000
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/gmmbc_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/gmmbc_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export WANDB_MODE=${WANDB_MODE:-online}
DATA=/coc/flash7/paphiwetsa3/datasets/circle_3000
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# GMM-BC (chunk8, PPO-compatible) on the FULL circle_3000 (3040 eps, VARIED goal).
# Same model as the 200-ep fixed-goal BC; the GMM head is goal-conditioned via the
# rendered image, so varied-goal training just teaches "push to the shown goal".
# c3000 yields >=50 packed batches/epoch, so the default tx_cos warmup->cosine
# scheduler (max_steps=90000=1800x50) is correct -- NO scheduler override (unlike
# the 200-ep set). eval_hnet_sim (random-goal) is IN-DISTRIBUTION here.
NAME=${NAME:-gmmBC_c3000}; DESC=gmm_bc_c3000
EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}
LOGGER=${LOGGER:-wandb}; MAXSTEPS=${MAXSTEPS:-400}
WANDBPROJ=""; [ "$LOGGER" = "wandb" ] && WANDBPROJ="logger.wandb.project=gmmbc_c3000"

LOGDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/${NAME}
RESUME=""
LAST=$(ls -t ${LOGDIR}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "$LAST" ]; then RESUME="ckpt_path='$LAST'"; echo "RESUMING from $LAST"; fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_tx_chunk8 \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
  evaluator.max_steps=${MAXSTEPS} evaluator.coverage_threshold=0.8 \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.precision=32 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  +trainer.gradient_clip_val=150 \
  trainer.profiler=null logger=${LOGGER} \
  ${WANDBPROJ} \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_dataloader_params.pushshapes_sim.num_workers=12 \
  +data.train_dataloader_params.pushshapes_sim.persistent_workers=true \
  +data.train_dataloader_params.pushshapes_sim.prefetch_factor=4 \
  +data.train_dataloader_params.pushshapes_sim.pin_memory=true \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.num_workers=4 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
  ${SEEDS:-} \
  $RESUME
echo "TRAIN_EXIT=$?"
