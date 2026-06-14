#!/bin/bash
#SBATCH --job-name=bcrnn_minmax_crop
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=8:00:00
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
# NOTE: deliberately DO NOT reuse the 5-episode precomputed norm_stats.json here.
# For norm_mode=minmax the stored min/max MUST bound the FULL training data for
# targets to land exactly in [-1,1]. The 5-ep stats (min=[0,4.5], max=[510,511.5])
# undershoot the full-data range (min=[0,0], max=[511.5,511.5]) -> ~8.8% of frames
# fall outside [-1,1]. With sample_frac=1.0 (config default) and no precomputed
# path, trainHydra computes true full-data min/max -> exact [-1,1].

# MIN-MAX action normalization (robomimic-faithful): maps [min,max] -> [-1,+1]
# exactly, so the GMM head's tanh'd mode means can represent every action target
# (quantile mode pushed ~2% of tail/edge actions outside [-1,1]).
SMOKE=${SMOKE:-0}
DESC=${DESC:-bc_rnn_minmax_crop}
NAME=${NAME:-bcrnn_minmax_crop_nc3}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnn_minmax_crop_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  EPOCHS=600; VALEVERY=100; LTB=8; LOGGER=wandb; MAXSTEPS=400; SEEDS=""; export WANDB_MODE=online
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_minmax_crop \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=ar \
  evaluator.max_steps=${MAXSTEPS} evaluator.coverage_threshold=0.8 ${SEEDS} \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax
echo "TRAIN_EXIT=$?"
