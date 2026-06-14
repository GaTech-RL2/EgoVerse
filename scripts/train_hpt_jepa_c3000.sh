#!/bin/bash
#SBATCH --job-name=hptFlowC3000Act8Jepa
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hptflow_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hptflow_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export WANDB_ENTITY=rl2-group WANDB_MODE=online

# Action-conditioned JEPA (Flavor B) on HPT act8, circle_3000. Direct A/B vs the
# baseline hptFlowC3000CausalAct8 (SAME data/hparams) -- the ONLY differences are
# model=...jepa (adds the JEPA aux, default-off elsewhere), data=...jepa (adds the
# future-obs target key), and lambda=50. Reuses the act32 c3000 norm (same data;
# action norm is per-dim, horizon-independent) -> skips the slow norm pass.
# Loader memory bounded for the 9-frame future window: batch 128 / prefetch 2.
NAME=hptFlowC3000Act8Jepa
# Reuse the act8 c3000 norm (horizon-8 action stats -> matches this act8 run).
# The act32 norm has horizon-32 action stats and triggers an 8-vs-32 mismatch in
# _apply_norm_one. Smoke-proven with this exact data config.
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC3000CausalAct8/hpt_flow_circle_3000_causal_obs1_act8_2026-06-14_01-39-26/norm_stats/norm_stats.json

RESUME=""
LAST=$(ls -t /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/${NAME}/*/checkpoints/last.ckpt 2>/dev/null | head -1)
[ -n "$LAST" ] && RESUME="ckpt_path=$LAST"

srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=${NAME} description=hpt_flow_circle_3000_act8_jepa_lambda50 mode=train \
  data=tsimulation_hpt_causal_jepa model=hpt_pushshapes_circle_jepa \
  ++model.robomimic_model.jepa.lambda=50 \
  model.optimizer.lr=4e-5 ~model.scheduler \
  +model.scheduler._target_=egomimic.utils.schedulers.warmup_cosine_scheduler \
  +model.scheduler._partial_=true +model.scheduler.max_steps=90000 \
  +model.scheduler.warmup_steps=500 +model.scheduler.warmup_start_factor=0.1 +model.scheduler.eta_min=4.0e-6 \
  evaluator=eval_hpt_standard \
  +evaluator.rollout_mode=chunk_openloop +evaluator.chunk_k=8 +evaluator.temporal_ensemble=false \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=100 \
  trainer=debug trainer.max_epochs=1800 trainer.limit_train_batches=50 \
  trainer.check_val_every_n_epoch=100 trainer.limit_val_batches=4 \
  trainer.profiler=null logger=csv_wandb logger.wandb.project=zarr_test \
  data.train_dataloader_params.pushshapes_sim.batch_size=128 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=128 \
  data.train_dataloader_params.pushshapes_sim.num_workers=8 \
  data.valid_dataloader_params.pushshapes_sim.num_workers=8 \
  data.train_dataloader_params.pushshapes_sim.prefetch_factor=2 \
  data.valid_dataloader_params.pushshapes_sim.prefetch_factor=2 \
  norm_stats.norm_mode=quantile \
  norm_stats.precomputed_norm_path=$NORM \
  ${RESUME}
echo "TRAIN_EXIT=$?"
