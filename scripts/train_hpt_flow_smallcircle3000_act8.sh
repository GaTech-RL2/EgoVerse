#!/bin/bash
#SBATCH --job-name=hptFlowSmallCircle3000Act8
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hptflow_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hptflow_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export WANDB_ENTITY=rl2-group

# BC small_circle_3000 baseline for HPT flow. Faithful clone of train_hpt_flow_c3000causal_act8.sh:
# ONLY dataset (circle_3000 -> small_circle_3000) + name/desc changed. Fresh quantile norm (no precomputed).
DS=/coc/flash7/paphiwetsa3/datasets/small_circle_3000

SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
  NAME=hptFlowSmallCircle3000Act8_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LOGGER=csv; PROJ=""; SEEDS="+evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  NAME=hptFlowSmallCircle3000Act8; EPOCHS=1800; VALEVERY=100; LTB=50; LOGGER=csv_wandb; PROJ="logger.wandb.project=zarr_test"; SEEDS=""; export WANDB_MODE=online
fi
DESC=hpt_flow_small_circle_3000_causal_obs1_act8

RESUME=""
if [ "$SMOKE" != "1" ]; then
  LAST=$(ls -t /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/${NAME}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
  [ -n "$LAST" ] && RESUME="ckpt_path=$LAST"
fi

srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train \
  data=tsimulation_hpt model=hpt_pushshapes_circle \
  ++model.robomimic_model.head_specs.pushshapes_sim.action_horizon=8 \
  ++model.robomimic_model.head_specs.pushshapes_sim.model.act_seq=8 \
  model.optimizer.lr=4e-5 ~model.scheduler \
  +model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  +model.scheduler._partial_=true +model.scheduler.max_steps=90000 \
  +model.scheduler.warmup_steps=500 +model.scheduler.warmup_start_factor=0.1 +model.scheduler.eta_min=4.0e-6 \
  evaluator=eval_hpt_standard \
  +evaluator.rollout_mode=chunk_openloop +evaluator.chunk_k=8 +evaluator.temporal_ensemble=false ${SEEDS} \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.max_epochs=${EPOCHS} trainer.limit_train_batches=${LTB} \
  trainer.check_val_every_n_epoch=${VALEVERY} trainer.limit_val_batches=4 \
  trainer.profiler=null logger=${LOGGER} ${PROJ} \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
  data.train_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
  data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
  data.train_dataloader_params.pushshapes_sim.batch_size=256 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=256 \
  data.train_dataloader_params.pushshapes_sim.num_workers=12 \
  data.valid_dataloader_params.pushshapes_sim.num_workers=12 \
  +data.train_dataloader_params.pushshapes_sim.pin_memory=true \
  +data.valid_dataloader_params.pushshapes_sim.pin_memory=true \
  +data.train_dataloader_params.pushshapes_sim.persistent_workers=true \
  +data.valid_dataloader_params.pushshapes_sim.persistent_workers=true \
  +data.train_dataloader_params.pushshapes_sim.prefetch_factor=4 \
  +data.valid_dataloader_params.pushshapes_sim.prefetch_factor=4 \
  ${RESUME}
echo "TRAIN_EXIT=$?"
