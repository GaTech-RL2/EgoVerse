#!/bin/bash
#SBATCH --job-name=bcrnnHnetSmallCircle3000C8FH
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --time=03:55:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bcrnn_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bcrnn_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
DATA=/coc/flash7/paphiwetsa3/datasets/small_circle_3000
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# BC small_circle_3000 baseline for real H-Net. Faithful clone of train_bcrnnHnetC8FH_c3000v2.sh:
# ONLY dataset (circle_3000 -> small_circle_3000) + name/desc changed, plus eval pusher_shape=circle_small
# so the in-training closed-loop coverage scores the SMALL pusher (r=5) not the big one.
# Model: bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist; fresh minmax on small_circle_3000.

SMOKE=${SMOKE:-0}
NAME=${NAME:-bcrnnHnetSmallCircle3000C8FH}
DESC=${DESC:-bc_rnn_hnet_chunk8_fullhist_smallcircle3000}
if [ "$SMOKE" = "1" ]; then
  NAME=bcrnnHnetSmallCircle3000C8FH_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; PROJ=""; MAXSTEPS=20; SEEDS="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  EPOCHS=${EPOCHS:-1800}; VALEVERY=${VALEVERY:-100}; LTB=${LTB:-50}; LOGGER=csv_wandb; PROJ="logger.wandb.project=zarr_test"; MAXSTEPS=400; SEEDS=""; export WANDB_MODE=online
fi

RESUME=""
if [ "$SMOKE" != "1" ]; then
  MARK=logs/bcrnnHnetSmallCircle3000C8FH/.launched
  if [ -f "$MARK" ]; then
    LAST=$(ls -t /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetSmallCircle3000C8FH/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LAST" ] && RESUME="ckpt_path=$LAST"
  fi
  mkdir -p logs/bcrnnHnetSmallCircle3000C8FH && touch "$MARK"
fi

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=ar \
  evaluator.max_steps=${MAXSTEPS} evaluator.coverage_threshold=0.8 ${SEEDS} \
  ++evaluator.env_kwargs.pusher_shape=circle_small \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.precision=32 trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} ${PROJ} \
  data.train_dataloader_params.pushshapes_sim.batch_size=16 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=16 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
   ${RESUME}
echo "TRAIN_EXIT=$?"
