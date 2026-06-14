#!/bin/bash
#SBATCH --job-name=bcrnnTxC4FH_c3000v2
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bcrnn_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bcrnn_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
DATA=/coc/flash7/paphiwetsa3/datasets/circle_3000
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# bcrnnTxC4FH_c3000v2: FRESH full-history circle_3000 run. Cloned from the original
# new_circle_3 launcher recipe; ONLY the dataset folder_path + name changed.
# Model: bc_rnn_pushshapes_paperexact_tx_chunk4; chunk_len=4; FH override rnn_horizon=160.
# norm: minmax, NO precomputed path -> trainHydra computes FRESH minmax stats on
# circle_3000 (different data than new_circle_3, so reuse is wrong).

SMOKE=${SMOKE:-0}
NAME=${NAME:-bcrnnTxC4FH_c3000v2}
DESC=${DESC:-bc_rnn_tx_chunk4_fullhist_c3000}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnnTxC4FH_c3000v2_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=wandb; MAXSTEPS=400; SEEDS=""; export WANDB_MODE=online
fi

# Requeue-safe resume hook: on SLURM auto-requeue (scavenger preemption), pick up
# the newest last.ckpt for THIS run name so it does NOT restart at ep0. Gated on a
# marker file created on first launch, so the FIRST run is always fresh (never
# resumes a stale dir of the same name).
RESUME=""
if [ "$SMOKE" != "1" ]; then
  MARK=logs/bcrnnTxC4FH_c3000v2/.launched
  if [ -f "$MARK" ]; then
    LAST=$(ls -t logs/bcrnnTxC4FH_c3000v2/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LAST" ] && RESUME="ckpt_path=$LAST"
  fi
  mkdir -p logs/bcrnnTxC4FH_c3000v2 && touch "$MARK"
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_tx_chunk4 \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=ar \
  evaluator.max_steps=${MAXSTEPS} evaluator.coverage_threshold=0.8 ${SEEDS} \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.precision=32 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
  model.robomimic_model.rnn_horizon=160 model.robomimic_model.core_net.max_window=160 +model.robomimic_model.window_anchor=start ${RESUME}
echo "TRAIN_EXIT=$?"
