#!/bin/bash
#SBATCH --job-name=bcrnnHnetC8FHR
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

# BC-RNN-HNET-CHUNK8-FULLHIST-RATIO (variant A+ratio, bcrnnHnetC8FHR). IDENTICAL
# recipe to bcrnnHnetC8FH (train_bc_rnn_hnet_chunk8_fullhist.sh) with the H-Net
# PAPER RATIO LOSS ACTIVATED. SINGLE-KNOB vs bcrnnHnetC8FH: the model config
# bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio flips
# collect_ratio_loss false->true, so the BCRNN algo reads the HNetCore chunker's
# auxiliary ratio loss each forward (already weighted by ratio_loss_weight=0.03,
# target_compression_ratio=2.0) and the inherited HNet.compute_losses optimizes
# action_loss + ratio_loss -- the chunker's boundary router now LEARNS to
# compress toward the target. EVERYTHING ELSE is unchanged from bcrnnHnetC8FH:
# rnn_horizon=80 (80 obs-steps * obs_stride 8 = 640 env frames), window_anchor=
# start, obs_stride/chunk_len 8/8, fp32, no grad clip, repeat-pad unmasked
# windows, raw low-dim obs + ReLU image + no fusion MLP, no actor MLP, eval
# RANDOM crop v0.2, minmax norm fresh full-data stats, warmup->cosine LR 1e-4.
#
# norm_stats: minmax, NO precomputed_norm_path, sample_frac=1.0 (config default)
# -> trainHydra computes true full-data min/max so all targets land in [-1,1].
# (Matches the tx_chunk8 / paperexact_hnet template; the train_bc_rnn_{cos,base}.sh
# variants that DO pass precomputed_norm_path use the OLD circle dataset's stats,
# which do not apply to new_circle_3.)

# ----- Batch composition + budget -----
# WINDOWS-PER-BATCH CONFOUND (documented, not fixed): window_anchor=start yields
# EXACTLY ONE window per episode in the batch (<= batch_size=16 windows/optimizer
# step), vs the uniform mode's up-to-256-window cap. So this run sees far fewer
# window-gradients per optimizer step than bcrnnHnetC8 (which fills the 256 cap).
# EPOCHS/LTB are kept identical for wall-clock comparability; the effective
# window-gradient budget differs and is a known confound when comparing A vs B.
SMOKE=${SMOKE:-0}
DESC=${DESC:-bc_rnn_hnet_chunk8_fullhist_ratio}
NAME=${NAME:-bcrnnHnetC8FHR_nc3}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnnHnetC8FHR_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  # ----- Budget (same as bcrnnHnetC8FH; A40 fp32) -----
  # EPOCHS=1800 x LTB=50 = 90,000 optimizer steps. VALEVERY=100 -> 18 rollout
  # evals + 18 checkpoints across the run (+ 'last').
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=wandb; MAXSTEPS=400; SEEDS=""; export WANDB_MODE=online
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio \
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
  '~data.valid_datasets.pushshapes_sim' \
  '+data.valid_datasets.pushshapes_sim._target_=egomimic.rldb.zarr.zarr_dataset_packed.ZarrEpisodePackedDataset.from_resolver' \
  '+data.valid_datasets.pushshapes_sim.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver' \
  +data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  +data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  +data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=1024 \
  '+data.valid_datasets.pushshapes_sim.resolver.transform_list=null' \
  '+data.valid_datasets.pushshapes_sim.chunking=none' \
  +data.valid_datasets.pushshapes_sim.min_seq_len=64 \
  '+data.valid_datasets.pushshapes_sim.max_seq_len=null' \
  norm_stats.norm_mode=minmax
  # VALID DATASET FIX (2026-06-06 tsimulation.yaml dedup): the dedup turned
  # valid_datasets.pushshapes_sim into the whole-node interpolation
  # ${data.train_datasets.pushshapes_sim}. Two failure modes that bit us:
  #   1. Hydra forbids .resolver.* sub-overrides on an interpolation node
  #      (ConfigKeyError "Key 'resolver' is not in struct").
  #   2. Just dropping the valid overrides ISN'T enough: the original
  #      interpolation resolved to the stale /coc/cedarp-.../Tsim_datasets2/circle
  #      folder + get_keymap (NOT new_circle_3 + get_keymap_eval) -> "filters
  #      matched no episodes" at instantiate time.
  # FIX: DELETE the stale interpolation (~) and RE-ADD valid as an EXPLICIT
  # literal struct (+) mirroring the OVERRIDDEN train node (new_circle_3 +
  # get_keymap_eval, chunking=none, min_seq_len=64) -- no interpolation, no
  # timing ambiguity. Matches exactly what the twin run 3325575 resolved to back
  # when valid was still an explicit struct (valid == train == new_circle_3 +
  # get_keymap_eval). (The 19 sibling BC launchers still carry the now-broken
  # .resolver.* valid overrides and will fail mode 1 identically until updated.)
echo "TRAIN_EXIT=$?"
