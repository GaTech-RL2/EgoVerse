#!/bin/bash
#SBATCH --job-name=hptFlowDpmatch
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --qos=long
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=omgwth,cyborg,ig-88,hk47,spd-13,sonny,kitt,cheetah,heistotron,megazord,puma,baymax,deebot,megabot
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hptflow_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hptflow_%x_%j.err
# Clone of train_hpt_flow_combo.sh (the original hptFlowCombo recipe) with:
#   model=hpt_bc_flow_pushshapes_dpmatch  (GroupNorm ResNet stem + RandomCrop86/CenterCrop86 augs)
#   callbacks=checkpoints_ema         (EMA shadow weights -> ema_state_dict in every ckpt)
#   name/description=hptFlowDpmatch
#   hoffman-lab -q long (7d, non-preemptible) instead of overcap scavenger
#   logger=csv_wandb for the real run (csv tail + same wandb project as before)
set -uxo pipefail
# slurm bin (srun) — needed when submitted from a non-interactive ssh shell,
# whose PATH (inherited by sbatch) lacks the slurm dir.
export PATH=/opt/slurm/Ubuntu-20.04/current/bin:$PATH
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export WANDB_ENTITY=rl2-group

# DATASET DEVIATION (2026-07-15): the original combo dataset circle_950norm_plus_new400
# is a husk of 1353 DANGLING symlinks — its targets (new_circle_3_normalized + flash7
# new_circle_3) were deleted in the Jul-10 datasets cleanup and no copy survives
# (checked flash7, scratch, cedarp, PACE, ZFS snapshots, S3). Using circle_3000
# (3040 eps, same embodiment/naming) — the family's standard since hptFlowC3000,
# which ran this exact recipe on it. Norm stats reused from that run's archive
# (quantile, computed over circle_3000).
DS=/coc/flash7/paphiwetsa3/datasets/circle_3000
NORM=${NORM:-}

SMOKE=${SMOKE:-0}
DESC=${DESC:-hptFlowDpmatch}
NAME=${NAME:-hptFlowDpmatch}
if [ "$SMOKE" = "1" ]; then
  NAME=hptFlowDpmatch_smoke; EPOCHS=2; VALEVERY=2; LTB=2; LVB=2; LOGGER=csv
  SCHED_MAX=12; SCHED_WARM=2; export WANDB_MODE=disabled
else
  EPOCHS=1800; VALEVERY=100; LTB=50; LVB=4; LOGGER=csv_wandb
  SCHED_MAX=90000; SCHED_WARM=500; export WANDB_MODE=online
fi

# Requeue-safe resume: if a prior run of this NAME left a last.ckpt, resume from the newest one
# (so a requeue auto-resumes instead of restarting at epoch 0).
EXTRA=()
[ -n "${NORM:-}" ] && EXTRA+=("norm_stats.precomputed_norm_path=${NORM}")
RESUME_CKPT="${CKPT:-}"
if [ -z "$RESUME_CKPT" ]; then
  LAST=$(ls -t /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/${NAME}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
  [ -n "$LAST" ] && RESUME_CKPT="$LAST"
fi
[ -n "$RESUME_CKPT" ] && EXTRA+=("ckpt_path='${RESUME_CKPT}'")

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train \
  data=tsimulation_hpt model=hpt_bc_flow_pushshapes_dpmatch \
  model.optimizer.lr=4e-5 \
  '~model.scheduler' \
  +model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  +model.scheduler._partial_=true \
  +model.scheduler.max_steps=${SCHED_MAX} \
  +model.scheduler.warmup_steps=${SCHED_WARM} \
  +model.scheduler.warmup_start_factor=0.1 \
  +model.scheduler.eta_min=4.0e-6 \
  evaluator=eval_hpt_standard \
  callbacks=checkpoints_ema callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=${LVB} \
  trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.train_datasets.pushshapes_sim.resolver.key_map.action_horizon=16 \
  data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=16 \
  ${EXTRA[@]+"${EXTRA[@]}"}
echo "TRAIN_EXIT=$?"
