#!/bin/bash
#SBATCH --job-name=bcrnnHnetCo
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
# COTRAIN MERGED FOLDER: a single directory of symlinks to BOTH circle datasets
#   big   = /coc/flash7/paphiwetsa3/datasets/new_circle_3        (953 .zarr eps; dir also holds a stray norm_stats.json + training_logs that the resolver skips)
#   small = /coc/flash7/paphiwetsa3/datasets/new_circle_small__3 (955 .zarr eps; pusher_shape=circle_small, embodiment=pushshapes_sim — SAME tag as big)
# => 953 + 955 = 1908 episodes served under ONE pushshapes_sim entry.
# WHY a merged folder (vs a 2nd train_datasets dict entry): the norm-stat
# inferencer is keyed by EMBODIMENT (get_embodiment_id(dataset_name)), and both
# datasets share embodiment "pushshapes_sim". A 2nd dict entry would (a) need a
# valid enum name as its key — there is no PUSHSHAPES_SIM_SMALL enum — and (b)
# OVERWRITE norm_stats[15] on the 2nd infer_norm_from_dataset call, so stats
# would reflect only ONE dataset, NOT the union. One merged folder => ONE
# resolver => ONE infer_norm_from_dataset over all 1908 eps => TRUE UNION minmax
# stats, and the model's domains/ac_keys (keyed by "pushshapes_sim") already
# cover both with NO model-config change. This is the single-knob-vs-3325170
# change: ONLY data (NC3 -> CO_MERGED).
CO_MERGED=/coc/flash7/paphiwetsa3/datasets/circle_co_big_small
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# BC-RNN-HNET COTRAIN. IDENTICAL recipe to train_bc_rnn_hnet.sh (job 3325170:
# same model=bc_rnn_pushshapes_paperexact_hnet, same budget EPOCHS/LTB, fp32
# trainer.precision=32, no grad clip, repeat-pad unmasked windows, raw low-dim
# obs + ReLU image + no fusion MLP, no actor MLP, eval RANDOM crop with v0.2
# off-by-one, minmax norm fresh full-data stats, same warmup->cosine schedule)
# -- the ONLY change is the DATA: the merged big+small circle folder instead of
# the big-only new_circle_3. This is the MVP cotrain, single-knob vs 3325170.
#
# norm_stats: minmax, NO precomputed_norm_path, sample_frac=1.0 (config default)
# -> trainHydra computes true full-data min/max over the UNION of both datasets
# so all targets land in [-1,1].
#
# IN-TRAINING EVAL IS BIG-CIRCLE-ONLY (accepted MVP limitation). The closed-loop
# sim evaluator (eval_hnet_sim) spawns PushShapesEnv with pusher_shape="circle"
# (big). The env hard-rejects any pusher_shape not in ("circle","stick") — there
# is NO "circle_small" pusher geometry in Tsimulation/pushshapes/env.py. So the
# small-circle variant CANNOT be closed-loop-evaluated in-training without a deep
# env port (new pusher radius/geometry), which is explicitly out of scope for
# this MVP. The in-training coverage number therefore measures the big-circle
# env only; offline small-env eval is FOLLOW-UP WORK. Training still learns from
# BOTH datasets (union batches + union norm stats) — only the eval is big-only.

# ----- Batch composition + budget (identical to 3325170) -----
# Packed pipeline serves full episodes which the algo cuts into 10-step windows
# (repeat-pad in paper-exact mode). batch_size=16 packed episodes;
# max_windows_per_batch=256 windows/step. STEPS/epoch = limit_train_batches.
SMOKE=${SMOKE:-0}
DESC=${DESC:-bc_rnn_hnet_cotrain}
NAME=${NAME:-bcrnnHnetCo_nc3}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnnHnetCo_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  # ----- Budget (same as 3325170; A40 fp32) -----
  # EPOCHS=1800 x LTB=50 = 90,000 optimizer steps. Each step trains up to 256
  # windows (max_windows_per_batch). VALEVERY=100 -> 18 rollout evals + 18
  # checkpoints across the run (+ 'last').
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=wandb; MAXSTEPS=400; SEEDS=""; export WANDB_MODE=online
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_hnet \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=ar \
  evaluator.max_steps=${MAXSTEPS} evaluator.coverage_threshold=0.8 ${SEEDS} \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.precision=32 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$CO_MERGED \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$CO_MERGED \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax
echo "TRAIN_EXIT=$?"
