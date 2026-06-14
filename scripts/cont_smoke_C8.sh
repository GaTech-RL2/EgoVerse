#!/bin/bash
#SBATCH --job-name=bcrnnTxC8cont_smoke
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1
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
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# --- GENTLE-TAIL CONTINUATION SMOKE for bcrnnTxC8_nc3 ---
# init_ckpt weights-only load of the orig run's latest epoch_epoch ckpt, then
# +1800 fresh epochs at FLAT constant LR=1e-6, fp32. SMOKE=2 epochs, csv logger.
ORIG=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxC8_nc3/bc_rnn_tx_chunk8_2026-06-05_23-40-22
INIT_CKPT=$ORIG/checkpoints/epoch_epoch=1599.ckpt
NORM=$ORIG/norm_stats/norm_stats.json

NAME=bcrnnTxC8_nc3_cont_smoke
DESC=bc_rnn_tx_chunk8_cont
EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"
export WANDB_MODE=disabled

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_tx_chunk8 \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=ar \
  evaluator.max_steps=${MAXSTEPS} evaluator.coverage_threshold=0.8 ${SEEDS} \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.precision=32 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
  norm_stats.precomputed_norm_path=$NORM \
  model.scheduler._target_=egomimic.utils.schedulers.constant_scheduler \
  '~model.scheduler.max_steps' '~model.scheduler.warmup_steps' \
  '~model.scheduler.warmup_start_factor' '~model.scheduler.eta_min' \
  ++model.optimizer.lr=1e-6 \
  +init_ckpt="'$INIT_CKPT'"
echo "TRAIN_EXIT=$?"
