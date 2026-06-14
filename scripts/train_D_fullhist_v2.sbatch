#!/bin/bash
#SBATCH --job-name=chunkD_fullhist_v2
#SBATCH --time=2-00:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/chunkD_fullhist_v2_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/chunkD_fullhist_v2_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
MODEL=hnet_pushshapes_fused_windowed_crossattn_resnet
# D-fullhist-v2 = D-fullhist (full obs-history, n_obs_history=16) + the NEW
# ++model.robomimic_model.hist_spatial_mode=per_frame knob. SINGLE knob vs
# D-fullhist: per_frame makes EVERY history frame processed IDENTICALLY -- each
# history cond token c_i cross-attends over ITS OWN frame i's M spatial tokens
# (the trailing BOS reads the CURRENT frame N-1), instead of D-fullhist's
# "current" mode where ALL tokens read only the current frame's spatial set and
# the past frames contribute only their pooled fused_cond "gist". This tests
# whether D-fullhist's regression (ep-200: 0.041 vs D's 0.120) was because the
# PAST was POOLED (spatial detail discarded) or because history is inherently
# useless here. The N>1 cross-attn wiring degenerates to D at N=1 (proven
# torch.equal: per_frame==current==plain-D). Everything else (K schedule,
# MAXW=16, MWS=96, GMM 5, flat 1e-4, 600ep, eval/200, norm_stats) is identical
# to train_D_fullhist.sbatch for comparability.
#
# TRAIN/EVAL ASYMMETRY (documented, same as D-fullhist): training samples chains
# at strided anchors and set_state's the sim to the ANCHOR -> training history
# reaches CHAIN start (<=6 frames deep). Eval resets to episode frame 0 -> eval
# history reaches EPISODE start (<=13 frames deep). Demo-prepend was REJECTED
# (would glue demo-rendered seams onto an all-sim chain the eval never has).
KSCHED='[[0,1],[100,2],[250,4],[400,6]]'
DESC=chunkD_fullhist_v2
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
  NAME=hnet_smoke; EPOCHS=2; VALEVERY=2; LOGGER=csv; MAXSTEPS=20
  EXTRA="evaluator.init_seeds=[0,1]"; export WANDB_MODE=disabled
else
  NAME=hnet_baseline_nc3; EPOCHS=600; VALEVERY=200; LOGGER=wandb; MAXSTEPS=400
  EXTRA=""; export WANDB_MODE=online
fi
MWS=96; MAXW=16; CHUNK_SIZE=32; N_OBS_HISTORY=16
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
  ++model.robomimic_model.n_obs_history=${N_OBS_HISTORY} \
  ++model.robomimic_model.hist_spatial_mode=per_frame \
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
