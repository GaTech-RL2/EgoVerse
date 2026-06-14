#!/bin/bash
#SBATCH --job-name=bcrnnTxC8FH
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
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# BC-RNN-TRANSFORMER-CHUNK8-FULLHIST (the bcrnnTxC8FH run). The exact Transformer
# TWIN of bcrnnHnetC8FH (train_bc_rnn_hnet_chunk8_fullhist.sh): SAME recipe as
# bcrnnTxC8 (train_bc_rnn_tx_chunk8.sh) with FULL-EPISODE HISTORY. The model
# config bc_rnn_pushshapes_paperexact_tx_chunk8_fullhist sets rnn_horizon=80
# (80 obs-steps * obs_stride 8 = 640 env frames of context, covering the longest
# 600-frame demo AND the 400-step rollout) + window_anchor=start (training windows
# anchored at episode frame 0, so train context == a never-reset rollout). fp32,
# no grad clip, repeat-pad unmasked windows, raw low-dim obs + ReLU image + no
# fusion MLP, no actor MLP, eval RANDOM crop v0.2, minmax norm fresh full-data
# stats, warmup->cosine LR peak 1e-4 -- all identical to bcrnnTxC8. The ONLY
# core-shape effect of rnn_horizon 10->80 is the TransformerCore's LEARNED
# positional table growing 10->80 rows (+70 rows, ~+31k params); no other shape
# changes (state_dict key-set identical).
#
# norm_stats: minmax, NO precomputed_norm_path, sample_frac=1.0 (config default)
# -> trainHydra computes true full-data min/max so all targets land in [-1,1].
# (Matches the tx_chunk8 / paperexact template; the train_bc_rnn_{cos,base}.sh
# variants that DO pass precomputed_norm_path use the OLD circle dataset's stats,
# which do not apply to new_circle_3.)

# ----- Batch composition + budget -----
# WINDOWS-PER-BATCH CONFOUND (documented, not fixed): window_anchor=start yields
# EXACTLY ONE window per episode in the batch (<= batch_size=16 windows/optimizer
# step), vs the uniform mode's up-to-256-window cap. So this run sees far fewer
# window-gradients per optimizer step than bcrnnTxC8 (which fills the 256 cap).
# EPOCHS/LTB are kept identical for wall-clock comparability; the effective
# window-gradient budget differs and is a known confound when comparing A vs B.
SMOKE=${SMOKE:-0}
DESC=${DESC:-bc_rnn_tx_chunk8_fullhist}
NAME=${NAME:-bcrnnTxC8FH_nc3}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnnTxC8FH_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  # ----- Budget (same as bcrnnTxC8; A40 fp32) -----
  # EPOCHS=1800 x LTB=50 = 90,000 optimizer steps. VALEVERY=100 -> 18 rollout
  # evals + 18 checkpoints across the run (+ 'last').
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=wandb; MAXSTEPS=400; SEEDS=""; export WANDB_MODE=online
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
  trainer.profiler=null logger=${LOGGER} \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax
echo "TRAIN_EXIT=$?"
