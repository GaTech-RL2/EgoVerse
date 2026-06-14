#!/bin/bash
#SBATCH --job-name=bcrnnHnetC32
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ig-88
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

# BC-RNN-HNET-CHUNK32 (LINEAR, obs_stride=32/chunk_len=32).
# SINGLE-KNOB clone of train_bc_rnn_hnet_chunk8.sh: same budget (EPOCHS/LTB), fp32
# trainer.precision=32, no grad clip, repeat-pad unmasked windows, raw low-dim obs
# + ReLU image + no fusion MLP, no actor MLP, eval RANDOM crop, warmup->cosine LR.
# The ONLY change is model=bc_rnn_pushshapes_paperexact_hnet_chunk32 (obs_stride/chunk_len 8 -> 32, LINEAR chunk head). Core stays the REAL
# dynamic-chunking H-Net (HNetCore, d_model=256, causal).
#
# norm_stats: minmax, NO precomputed_norm_path, sample_frac=1.0 (config default)
# -> trainHydra computes FRESH full-data min/max on new_circle_3 so all targets
# land in [-1,1]. (No precomputed path -- the hnet/chunk family computes fresh
# stats; the precomputed-path variants use OLD circle stats that do not apply.)

SMOKE=${SMOKE:-0}
DESC=${DESC:-bc_rnn_hnet_chunk32}
NAME=${NAME:-bcrnnHnetC32_nc3}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnnHnetC32_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=wandb; MAXSTEPS=400; SEEDS=""; export WANDB_MODE=online
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_hnet_chunk32 \
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
