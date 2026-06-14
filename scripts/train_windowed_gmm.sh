#!/bin/bash
#SBATCH --job-name=windowed
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/windowed_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/windowed_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-online}   # default online; pre-set (e.g. offline) for smokes
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
OBS=${OBS:-demo}; EPOCHS=${EPOCHS:-600}; VALEVERY=${VALEVERY:-100}; DESC=${DESC:-windowed}
MODEL=${MODEL:-hnet_pushshapes_fused_windowed}   # _resnet variant -> ResNet vision encoder
MAXW=${MAXW:-128}; SMOKE=${SMOKE:-0}; CKPT=${CKPT:-}   # CKPT set -> resume fit() from it
MWS=${MWS:-0}; GCLIP=${GCLIP:-}                         # MWS=max_window_steps budget; GCLIP=grad clip val
REACTIVE=${REACTIVE:-}                                  # reactive=true -> current-frame-only (no obs/action history)
LR=${LR:-}                                              # override optimizer lr (lower -> smoother/lower loss)
CHUNK_ROLLOUT=${CHUNK_ROLLOUT:-}                        # chunk-B: true -> chunked closed-loop rollout trainer
CHUNK_SIZE=${CHUNK_SIZE:-32}                            # chunk-B: frames per chunk (also sets chunk_k head width)
DAGGER=${DAGGER:-}                                      # chunk-B+sim: true -> DAgger expert relabeling (supervise vs scripted expert at visited states)
N_OBS_HISTORY=${N_OBS_HISTORY:-1}                       # OBS-HISTORY window: N>1 -> condition each chunk on the last N obs frames (obs-only, no past actions). 1 = single-frame (byte-identical default)
ROLLOUT_MODE=${ROLLOUT_MODE:-ar}                        # eval rollout mode: ar | chunk_te | chunk_openloop
EVAL_CHUNK_K=${EVAL_CHUNK_K:-32}                        # eval chunk length for chunk_te/chunk_openloop
ACTION_HEAD=${ACTION_HEAD:-}                            # gmm -> GMM action head (NLL loss); unset -> continuous (byte-identical to train_windowed.sh)
GMM_MODES=${GMM_MODES:-5}                               # GMM mixture components (M); only used when ACTION_HEAD=gmm
# k_schedule defined IN-SCRIPT: its commas can't survive SLURM --export (comma-delimited),
# which truncated it to malformed '[[0,2]' and crashed Hydra. SMOKE flag has no commas.
KFIX=${KFIX:-}                                             # KFIX=N -> fixed K=N always (no curriculum)
KSCHED_ENV=${KSCHED:-}                                      # KSCHED env override -> use verbatim (set it from an sbatch, NOT --export: commas survive env, not --export)
if [ -n "$KSCHED_ENV" ]; then
  KSCHED="$KSCHED_ENV"                                      # caller-supplied custom schedule (e.g. dagger curriculum)
elif [ -n "$KFIX" ]; then
  KSCHED="[[0,$KFIX]]"                                      # fixed K=$KFIX every epoch (one-by-one if 1)
elif [ "$SMOKE" = "1" ]; then
  KSCHED='[[0,2],[2,8]]'                                    # short: K=2 (ep0-1) -> K=8 (ep2+)
elif [ "$SMOKE" = "2" ]; then
  KSCHED='[[0,32]]'                                         # K=32 from epoch 0 (OOM/budget smoke)
else
  KSCHED='[[0,1],[50,2],[100,4],[150,8],[250,16],[400,32]]' # full curriculum
fi
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=hnet_baseline_nc3 description=${DESC} mode=train data=tsimulation \
  model=${MODEL} \
  ${CKPT:+++ckpt_path=$CKPT} \
  ++model.robomimic_model.obs_source=${OBS} \
  ++model.robomimic_model.k_schedule="${KSCHED}" \
  ++model.robomimic_model.max_windows=${MAXW} \
  ++model.robomimic_model.max_window_steps=${MWS} \
  ${CHUNK_ROLLOUT:+++model.robomimic_model.chunk_rollout=$CHUNK_ROLLOUT} \
  ${CHUNK_ROLLOUT:+++model.robomimic_model.chunk_size=$CHUNK_SIZE} \
  ${CHUNK_ROLLOUT:+++model.robomimic_model.chunk_k=$CHUNK_SIZE} \
  ${ACTION_HEAD:+++model.robomimic_model.action_head=$ACTION_HEAD} \
  ${ACTION_HEAD:+++model.robomimic_model.gmm_num_modes=$GMM_MODES} \
  ${DAGGER:+++model.robomimic_model.dagger=$DAGGER} \
  ++model.robomimic_model.n_obs_history=${N_OBS_HISTORY} \
  ${REACTIVE:+++model.robomimic_model.reactive=$REACTIVE} \
  ${LR:+++model.optimizer.lr=$LR} \
  ${GCLIP:++trainer.gradient_clip_val=$GCLIP} \
  +callbacks.kcurriculum._target_=egomimic.algo.hnet_closedloop.KCurriculumCallback \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=${ROLLOUT_MODE} evaluator.chunk_k=${EVAL_CHUNK_K} \
  callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=${VALEVERY} \
  trainer=debug trainer.max_epochs=${EPOCHS} trainer.min_epochs=${EPOCHS} \
  trainer.limit_train_batches=8 trainer.limit_val_batches=4 trainer.check_val_every_n_epoch=${VALEVERY} \
  trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  norm_stats.precomputed_norm_path=$NORM
echo "TRAIN_EXIT=$?"
