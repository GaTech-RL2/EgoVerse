#!/bin/bash
#SBATCH --job-name=bcrnnHnetSmallC8FHR
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
# SMALL-CIRCLE clone of train_bc_rnn_hnet_chunk8_fullhist_ratio.sh (bcrnnHnetC8FHR).
# ONLY the dataset changes vs the big-circle FHR run: new_circle_3 -> new_circle_small__3
# (955 small-circle episodes, SAME schema: front_img_1 + state_agent_obj, 2-d cursor
# actions, 96x96). Single embodiment (pushshapes_sim), no routing. Norm is computed
# FRESH minmax over the small dataset (norm_stats.norm_mode=minmax, no precomputed path).
NCS=/coc/flash7/paphiwetsa3/datasets/new_circle_small__3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# BC-RNN-HNET-CHUNK8-FULLHIST-RATIO (variant A+ratio, bcrnnHnetSmallC8FHR_nc3). IDENTICAL
# recipe to the big-circle bcrnnHnetC8FHR (train_bc_rnn_hnet_chunk8_fullhist_ratio.sh)
# with the H-Net PAPER RATIO LOSS ACTIVATED. The model config
# bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio flips
# collect_ratio_loss false->true, so the BCRNN algo reads the HNetCore chunker's
# auxiliary ratio loss each forward (already weighted by ratio_loss_weight=0.03,
# target_compression_ratio=2.0) and the inherited HNet.compute_losses optimizes
# action_loss + ratio_loss -- the chunker's boundary router LEARNS to compress
# toward the target. EVERYTHING ELSE is unchanged: rnn_horizon=80 (80 obs-steps *
# obs_stride 8 = 640 env frames), window_anchor=start, obs_stride/chunk_len 8/8,
# fp32, no grad clip, repeat-pad unmasked windows, raw low-dim obs + ReLU image +
# no fusion MLP, no actor MLP, minmax norm fresh full-data stats, warmup->cosine
# LR 1e-4.
#
# norm_stats: minmax, NO precomputed_norm_path, sample_frac=1.0 (config default)
# -> trainHydra computes true full-data min/max OVER THE SMALL DATASET so all
# targets land in [-1,1]. (Fresh stats -- the big-circle stats are NOT reused.)
#
# EVAL CHANGE vs big-circle FHR (the ONE deviation forced by the env gap):
# evaluator=eval_hnet (HNetEvalVideo, teacher-forced val MSE + GT-vs-pred overlay,
# NO PushShapesEnv) INSTEAD OF evaluator=eval_hnet_sim. The closed-loop sim env
# (Tsimulation.pushshapes.PushShapesEnv) hardcodes PUSHER_RADIUS=15.0 / pusher_shape
# "circle" with NO circle_small geometry, so an in-training sim rollout on this run
# would eval the WRONG pusher and is dropped to keep the job from crashing on eval.
# Training + teacher-forced val loss/overlay remain fully valid for circle_small.
# (The big-circle launcher's evaluator.rollout_mode/max_steps/coverage_threshold/
# init_seeds overrides are dropped here: HNetEvalVideo takes none of them.)

# ----- Batch composition + budget -----
# WINDOWS-PER-BATCH CONFOUND (documented, not fixed): window_anchor=start yields
# EXACTLY ONE window per episode in the batch (<= batch_size=16 windows/optimizer
# step). EPOCHS/LTB kept identical to the big-circle FHR run for comparability.
SMOKE=${SMOKE:-0}
DESC=${DESC:-bc_rnn_hnet_smallcircle_c8_fhr}
NAME=${NAME:-bcrnnHnetSmallC8FHR_nc3}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnnHnetSmallC8FHR_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; export WANDB_MODE=disabled
else
  # ----- Budget (same as bcrnnHnetC8FHR; A40 fp32) -----
  # EPOCHS=1800 x LTB=50 = 90,000 optimizer steps. VALEVERY=100 -> 18 TF-val
  # passes + 18 checkpoints across the run (+ 'last').
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=wandb; export WANDB_MODE=online
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio \
  evaluator=eval_hnet \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.precision=32 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NCS \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  '~data.valid_datasets.pushshapes_sim' \
  '+data.valid_datasets.pushshapes_sim._target_=egomimic.rldb.zarr.zarr_dataset_packed.ZarrEpisodePackedDataset.from_resolver' \
  '+data.valid_datasets.pushshapes_sim.resolver._target_=egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver' \
  +data.valid_datasets.pushshapes_sim.resolver.folder_path=$NCS \
  +data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  +data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=1024 \
  '+data.valid_datasets.pushshapes_sim.resolver.transform_list=null' \
  '+data.valid_datasets.pushshapes_sim.chunking=none' \
  +data.valid_datasets.pushshapes_sim.min_seq_len=64 \
  '+data.valid_datasets.pushshapes_sim.max_seq_len=null' \
  norm_stats.norm_mode=minmax
  # VALID DATASET FIX (inherited from the big-circle launcher; tsimulation.yaml
  # dedup turned valid_datasets.pushshapes_sim into a whole-node interpolation).
  # FIX: DELETE the stale interpolation (~) and RE-ADD valid as an EXPLICIT
  # literal struct (+) mirroring the OVERRIDDEN train node (new_circle_small__3 +
  # get_keymap_eval, chunking=none, min_seq_len=64) -- no interpolation.
echo "TRAIN_EXIT=$?"
