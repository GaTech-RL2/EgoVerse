#!/bin/bash
#SBATCH --job-name=bcrnnTxCosB200
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
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3_bal200
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# BC-RNN-TRANSFORMER on the BALANCED 200-episode subset (new_circle_3_bal200).
# IDENTICAL recipe to train_bc_rnn_tx_cos_200ep.sh; the ONLY change is the
# dataset folder: new_circle_3_first200 (naive "first 200 sorted") ->
# new_circle_3_bal200 (200 episodes chosen by k-center greedy / farthest-point
# sampling on normalized [Tstart_x,Tstart_y,goal_x,goal_y], obstacle-level-0
# only, for balanced joint coverage of the T's start pose and goal pose).
# See new_circle_3_bal200/SELECTION.md for method + evidence grids.
#
# model=bc_rnn_pushshapes_paperexact_tx_cos: recurrent core is a causal-attention
# TRANSFORMER (d_model=448, 5 layers, 8 heads).
#
# norm_stats: minmax, NO precomputed_norm_path, sample_frac=1.0 (config default)
# -> trainHydra computes FRESH true min/max over THIS subset so all targets
# land in [-1,1].

# ----- Batch composition + budget (identical to 200ep) -----
SMOKE=${SMOKE:-0}
DESC=${DESC:-bc_rnn_tx_cos_bal200}
NAME=${NAME:-bcrnnTxCosB200_nc3}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnnTxCosB200_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  # ----- Budget (same as 200ep; A40 fp32) -----
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=wandb; MAXSTEPS=400; SEEDS=""; export WANDB_MODE=online
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_tx_cos \
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
  norm_stats.norm_mode=minmax
echo "TRAIN_EXIT=$?"
