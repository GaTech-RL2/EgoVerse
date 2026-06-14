#!/bin/bash
#SBATCH --job-name=C8Qcont
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
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
export WANDB_MODE=online
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# ---- GENTLE-TAIL CONTINUATION of bcrnnTxC8Q_nc3 ----
# weights-only init_ckpt of orig latest epoch_epoch ckpt, +1800 fresh epochs,
# FLAT constant LR=1e-6, fp32. Same model+data group + norm_stats as orig.
ORIG=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxC8Q_nc3/bc_rnn_tx_chunk8_queries_2026-06-06_06-19-57
INIT_CKPT=$ORIG/checkpoints/epoch_epoch=1599.ckpt
NORM=$ORIG/norm_stats/norm_stats.json

NAME=bcrnnTxC8Q_nc3_cont
DESC=bc_rnn_tx_chunk8_queries_cont
EPOCHS=1800; VALEVERY=100; LTB=50; MAXSTEPS=400

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_tx_chunk8_q \
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
  model.scheduler._target_=egomimic.utils.schedulers.constant_scheduler \
  '~model.scheduler.max_steps' '~model.scheduler.warmup_steps' \
  '~model.scheduler.warmup_start_factor' '~model.scheduler.eta_min' \
  ++model.optimizer.lr=1e-6 \
  +init_ckpt="'$INIT_CKPT'"
echo "TRAIN_EXIT=$?"
