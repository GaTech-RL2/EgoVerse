#!/bin/bash
#SBATCH --job-name=bcrnnTxSmallC8
#SBATCH --exclude=ig-88
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bcrnn_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bcrnn_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
# SMALL-CIRCLE transformer chunk-8 WITH closed-loop COVERAGE eval (shows on wandb).
# Derived from the big-circle eval_hnet_sim launcher, retargeted to new_circle_small__3
# + the small pusher (++evaluator.env_kwargs.pusher_shape=circle_small, radius auto 5.0).
# model = bc_rnn_pushshapes_paperexact_tx_chunk8 (Transformer core). RESUME=<ckpt> to full-resume.
NCS=/coc/flash7/paphiwetsa3/datasets/new_circle_small__3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetSmallC8FHR_nc3/bc_rnn_hnet_smallcircle_c8_fhr_2026-06-07_16-11-07/norm_stats/norm_stats.json

SMOKE=${SMOKE:-0}
RESUME=${RESUME:-}   # set to a ckpt path to full-resume (restores optim/sched/epoch)
DESC=${DESC:-bc_rnn_tx_smallcircle_c8}
NAME=${NAME:-bcrnnTxSmallC8_nc3}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnnTxSmallC8_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=wandb; MAXSTEPS=400; SEEDS=""; export WANDB_MODE=online
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_tx_chunk8 \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=ar \
  evaluator.max_steps=${MAXSTEPS} evaluator.coverage_threshold=0.8 ${SEEDS} \
  ++evaluator.env_kwargs.pusher_shape=circle_small \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.precision=32 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NCS \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NCS \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=$NORM \
  ${RESUME:+ckpt_path=$RESUME}
echo "TRAIN_EXIT=$?"
