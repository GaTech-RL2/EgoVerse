#!/bin/bash
#SBATCH --job-name=smallCL
#SBATCH --time=00:50:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/smallCL_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/smallCL_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200

RUN=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetCo_nc3/bc_rnn_hnet_cotrain_2026-06-06_02-55-53
SMALL=/coc/flash7/paphiwetsa3/datasets/new_circle_small__3
NORM=$RUN/norm_stats/norm_stats.json
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
CKPT=$RUN/checkpoints/ep1200.ckpt
export ROLLOUT_VIDEO_DIR=$RUN/videos/small_closedloop

srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=smallCL description=small_circle_closedloop mode=eval data=tsimulation \
  ckpt_path=$CKPT \
  model=bc_rnn_pushshapes_paperexact_hnet \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
  evaluator.init_mode=replay evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  ++evaluator.env_kwargs.pusher_shape=circle_small \
  ++evaluator.env_kwargs.object_shape=T ++evaluator.env_kwargs.image_size=96 \
  evaluator.limit_val_batches=6 trainer.limit_val_batches=6 \
  trainer=debug trainer.precision=32 trainer.profiler=null logger=wandb \
  data.train_dataloader_params.pushshapes_sim.batch_size=1 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=1 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$SMALL \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$SMALL \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.precomputed_norm_path=$NORM
echo "CL_EXIT=$?"
ls -la $ROLLOUT_VIDEO_DIR 2>/dev/null
