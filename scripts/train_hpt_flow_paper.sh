#!/bin/bash
#SBATCH --job-name=hptFlowPaper
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

# NEW dataset, no precomputed norm -> let it compute (norm_mode=quantile, config default).
DS=/coc/flash7/paphiwetsa3/datasets/pushshapes_paper

SMOKE=${SMOKE:-0}
DESC=${DESC:-hpt_flow_pushshapes_paper}
NAME=${NAME:-hptFlowPaper}
if [ "$SMOKE" = "1" ]; then
  NAME=hptFlowPaper_smoke; EPOCHS=2; VALEVERY=2; LTB=6; LVB=2; LOGGER=csv
  SCHED_MAX=12; SCHED_WARM=2; export WANDB_MODE=disabled
else
  EPOCHS=1800; VALEVERY=100; LTB=50; LVB=4; LOGGER=wandb
  SCHED_MAX=90000; SCHED_WARM=500; export WANDB_MODE=online
fi

# Optional: NORM=<norm_stats.json> skips the slow norm-stats recompute;
# CKPT=<path.ckpt> resumes training from a checkpoint (inner quotes guard '=' in filenames).
EXTRA=()
[ -n "${NORM:-}" ] && EXTRA+=("norm_stats.precomputed_norm_path=${NORM}")
# requeue-safe: if no explicit CKPT, auto-resume from this run's newest last.ckpt
# (so a SLURM preemption->requeue continues instead of restarting at ep0)
if [ -z "${CKPT:-}" ]; then CKPT=$(ls -t /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/${NAME}/*/checkpoints/last.ckpt 2>/dev/null | head -1); fi
[ -n "${CKPT:-}" ] && EXTRA+=("ckpt_path='${CKPT}'")

srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train \
  data=tsimulation_hpt model=hpt_pushshapes_circle \
  model.optimizer.lr=4e-5 \
  '~model.scheduler' \
  +model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  +model.scheduler._partial_=true \
  +model.scheduler.max_steps=${SCHED_MAX} \
  +model.scheduler.warmup_steps=${SCHED_WARM} \
  +model.scheduler.warmup_start_factor=0.1 \
  +model.scheduler.eta_min=4.0e-6 \
  evaluator=eval_hpt_standard \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=${LTB} trainer.limit_val_batches=${LVB} \
  trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS \
  ${EXTRA[@]+"${EXTRA[@]}"}
echo "TRAIN_EXIT=$?"
