#!/bin/bash
# H-Net baseline on new_circle_3 — teacher-forced packed training + AR sim eval.
# Usage: sbatch --export=ALL,HNET_MODEL=<model>,HNET_DESC=<desc> sbatch_hnet_baseline.sh
#   HNET_MODEL=hnet_pushshapes_fused  (nochunk / flat T8)   -> desc fused_nochunk
#   HNET_MODEL=hnet_pushshapes        (chunked / 3-stage)   -> desc chunked
#SBATCH --job-name=hnet-base
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hnet_base_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hnet_base_%x_%j.err
set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
mkdir -p logs/sbatch
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
hostname; nvidia-smi || true

export PYTHONPATH=.
export MUJOCO_GL=egl
export WANDB_MODE=online
export WANDB_ENTITY=rl2-group
export HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200

NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
# Fresh H-Net-keymap norm stats computed during the 5ep smoke (correct shape for
# packed per-frame actions; the HPT action_horizon=32 json is incompatible).
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
MODEL=${HNET_MODEL:-hnet_pushshapes_fused}
DESC=${HNET_DESC:-fused_nochunk}

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_baseline_nc3 description=${DESC}_80ep \
  mode=train \
  data=tsimulation \
  model=${MODEL} +model.robomimic_model.token_dropout_p=${HNET_TD:-0.0} ${HNET_EXTRA:-} \
  model.scheduler.max_steps=$(( ${HNET_EPOCHS:-80} * 8 )) \
  evaluator=eval_hnet_sim \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=20 \
  trainer=debug \
  trainer.max_epochs=${HNET_EPOCHS:-80} \
  trainer.min_epochs=${HNET_EPOCHS:-80} \
  trainer.limit_train_batches=8 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=20 \
  trainer.profiler=null \
  logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=null \
  evaluator.coverage_threshold=0.8 \
  norm_stats.precomputed_norm_path=$NORM
echo "EXIT_CODE=$?"
