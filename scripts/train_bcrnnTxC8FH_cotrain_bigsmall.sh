#!/bin/bash
#SBATCH --job-name=bcrnnTxC8FHCotrainBigSmall
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --exclude=ig-88
#SBATCH --requeue
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bcrnn_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bcrnn_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export WANDB_ENTITY=rl2-group
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
# SAME combined big+small folder as the HPT cotrain (mixed under one pushshapes_sim head)
DATA=/coc/flash7/paphiwetsa3/datasets/circle_3000_plus_small_circle_3000
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# BC-RNN transformer-core, chunk-8, full-history; cotrain on big+small (data-mix, one head).
# Recipe cloned verbatim from train_bcrnnTxC8FH_c3000v2.sh; only dataset/name/logger changed.
# norm: minmax (GMM head -> minmax, never quantile), NO precomputed -> fresh minmax on the mixed set.

SMOKE=${SMOKE:-0}
NAME=${NAME:-bcrnnTxC8FHCotrainBigSmall}
DESC=${DESC:-bc_rnn_tx_chunk8_fullhist_cotrain_big_small}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnnTxC8FHCotrainBigSmall_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; PROJ=""; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=csv_wandb; PROJ="logger.wandb.project=zarr_test"; MAXSTEPS=400; SEEDS=""; export WANDB_MODE=online
fi

# Requeue-safe resume hook: on SLURM auto-requeue (scavenger preemption), pick up the newest
# last.ckpt for THIS run name so it does NOT restart at ep0. Marker-gated so the FIRST run is fresh.
RESUME=""
if [ "$SMOKE" != "1" ]; then
  MARK=logs/bcrnnTxC8FHCotrainBigSmall/.launched
  if [ -f "$MARK" ]; then
    LAST=$(ls -t /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnTxC8FHCotrainBigSmall/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LAST" ] && RESUME="ckpt_path=$LAST"
  fi
  mkdir -p logs/bcrnnTxC8FHCotrainBigSmall && touch "$MARK"
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_tx_chunk8_fullhist \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=ar \
  evaluator.max_steps=${MAXSTEPS} evaluator.coverage_threshold=0.8 ${SEEDS} \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.precision=32 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} ${PROJ} \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  ++data.train_dataloader_params.pushshapes_sim.num_workers=16 \
  ++data.valid_dataloader_params.pushshapes_sim.num_workers=16 \
  ++data.train_dataloader_params.pushshapes_sim.pin_memory=true \
  ++data.valid_dataloader_params.pushshapes_sim.pin_memory=true \
  ++data.train_dataloader_params.pushshapes_sim.persistent_workers=true \
  ++data.valid_dataloader_params.pushshapes_sim.persistent_workers=true \
  ++data.train_dataloader_params.pushshapes_sim.prefetch_factor=4 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
   ${RESUME}
echo "TRAIN_EXIT=$?"
