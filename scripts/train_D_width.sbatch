#!/bin/bash
#SBATCH --job-name=chunkD_width
#SBATCH --time=2-00:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/chunkD_width_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/chunkD_width_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
MODEL=hnet_pushshapes_fused_windowed_crossattn_resnet
# D width ablation: SAME as train_D_crossattn.sbatch (run D, 3308556) except
# max_windows (window batch W). max_window_steps is raised to 6*W so the
# requested W actually holds at every K of the schedule (D's MWS=96 would
# clamp W back to 16 at K=6, making the ablation inert late in training).
# Memory: W*K^2 at K=6 = 36*W (W=64 -> 2304), far under the proven 32768.
KSCHED='[[0,1],[100,2],[250,4],[400,6]]'
MAXW=${MAXW:-32}
MWS=$((MAXW * 6))
DESC=chunkD_w${MAXW}
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
  NAME=hnet_smoke; EPOCHS=2; VALEVERY=2; LOGGER=csv; MAXSTEPS=20
  EXTRA="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  NAME=hnet_baseline_nc3; EPOCHS=600; VALEVERY=200; LOGGER=wandb; MAXSTEPS=400
  EXTRA=""; export WANDB_MODE=online
fi
CHUNK_SIZE=32
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=${NAME} description=${DESC} mode=train data=tsimulation \
  model=${MODEL} \
  ++model.robomimic_model.obs_source=sim \
  ++model.robomimic_model.k_schedule="${KSCHED}" \
  ++model.robomimic_model.max_windows=${MAXW} \
  ++model.robomimic_model.max_window_steps=${MWS} \
  ++model.robomimic_model.chunk_rollout=true \
  ++model.robomimic_model.chunk_size=${CHUNK_SIZE} \
  ++model.robomimic_model.chunk_k=${CHUNK_SIZE} \
  ++model.robomimic_model.action_head=gmm \
  ++model.robomimic_model.gmm_num_modes=5 \
  ++model.robomimic_model.n_obs_history=1 \
  +callbacks.kcurriculum._target_=egomimic.algo.hnet_closedloop.KCurriculumCallback \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=chunk_openloop evaluator.chunk_k=32 ${EXTRA} \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=8 trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=${LOGGER} \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=${MAXSTEPS} evaluator.coverage_threshold=0.8 \
  norm_stats.precomputed_norm_path=$NORM
echo "TRAIN_EXIT=$?"
