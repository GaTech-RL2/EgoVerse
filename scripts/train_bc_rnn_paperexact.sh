#!/bin/bash
#SBATCH --job-name=bcrnn_paperexact
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2/logs/sbatch/bcrnn_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2/logs/sbatch/bcrnn_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2
source /coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# BC-RNN PAPER-EXACT v2. fp32 everywhere (trainer.precision=32), no grad clip
# (enable_grad_norm=false in the config), repeat-pad unmasked windows, raw
# low-dim obs + ReLU image + no fusion MLP, RNN dim 1000, no actor MLP, eval
# RANDOM crop with v0.2 off-by-one. minmax norm (fresh full-data stats).
#
# norm_stats: minmax, NO precomputed_norm_path, sample_frac=1.0 (config default)
# -> trainHydra computes true full-data min/max so all targets land in [-1,1].

# ----- Batch composition + budget (audit item 8) -----
# No NATIVE i.i.d. windowed dataset mode is reachable through the packed config
# (ZarrEpisodePackedDataset.__getitem__ -> _read_span reads FULL spans; the
# per-frame horizon-windowed ZarrDataset path is not wrapped by the packed
# loader). Best feasible approximation: the packed pipeline serves full episodes
# which the algo cuts into 10-step windows (repeat-pad in paper-exact mode).
# batch_size=16 packed episodes; max_windows_per_batch=256 windows/step.
# STEPS/epoch = limit_train_batches (LTB). Budget chosen from the measured A40
# step time to fit a 48h overcap allocation (see report for the arithmetic and
# the residual deviation vs the paper's 300k steps).
SMOKE=${SMOKE:-0}
DESC=${DESC:-bc_rnn_paperexact}
NAME=${NAME:-bcrnn_paperexact_nc3}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnn_paperexact_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  # ----- Budget (measured A40 fp32 step time = ~1.70 s/step @ 0.59 it/s) -----
  # EPOCHS=1800 x LTB=50 = 90,000 optimizer steps ~= 42.4h pure train, leaving
  # ~5.6h of the 48h alloc for norm-stats compute (~3-4min), ~18 sim evals, and
  # checkpoint IO. Each step trains 256 windows (max_windows_per_batch) vs the
  # paper's batch of 16 -> our 90k steps = ~5.3x the paper's 300k-step
  # window-gradient throughput, so this MATCHES/EXCEEDS the paper budget even
  # though the optimizer-step count is lower (residual deviation: per-step
  # window batch is 256 not 16, and windows come from packed-episode cutting,
  # not a native i.i.d. SequenceDataset -- documented in the report).
  # VALEVERY=100 -> 18 rollout evals + 18 checkpoints across the run (+ 'last').
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=wandb; MAXSTEPS=400; SEEDS=""; export WANDB_MODE=online
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation/tsimulation \
  model=bc_rnn/pushshapes_paperexact \
  evaluator=hnet/sim \
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
