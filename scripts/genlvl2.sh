#!/bin/bash
#SBATCH --job-name=genlvl2
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=ig-88
#SBATCH --requeue
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/genlvl/%x_%A_%a.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/genlvl/%x_%A_%a.err
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
# Params via --export: CELL, RUN, DATASET, FAMILY (hpt|packed), MODEL (packed only)
# EXTRA: extra hydra overrides the run trained with but the base config lacks.
# C4 H-Net runs trained _hnet_chunk4 + full-history window args; the eval MUST replay
# them or core_net attends to the wrong window -> wrong actions -> spuriously low coverage.
EXTRA=${EXTRA:-}
case "$CELL" in
  hnetC4_*) [ -z "$EXTRA" ] && EXTRA="++model.robomimic_model.rnn_horizon=160 ++model.robomimic_model.core_net.max_window=160 ++model.robomimic_model.window_anchor=start" ;;
esac
LVL=${SLURM_ARRAY_TASK_ID:-0}
RD=$(ls -dt logs/${RUN}/*/ 2>/dev/null | head -1)
CK=${RD}checkpoints/last.ckpt
NORM=${RD}norm_stats/norm_stats.json
SEEDS='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]'
VID=logs/genlvl_vids/${CELL}/lvl${LVL}; mkdir -p "$VID"; export ROLLOUT_VIDEO_DIR=$VID
echo "CELL=$CELL FAMILY=$FAMILY LVL=$LVL RD=$RD VID=$VID"

run_hpt() {
  srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    name=g2_${CELL}_L${LVL} mode=eval data=tsimulation model=hpt_pushshapes_circle \
    ++model.robomimic_model.head_specs.pushshapes_sim.action_horizon=8 \
    ++model.robomimic_model.head_specs.pushshapes_sim.model.act_seq=8 \
    ckpt_path=${CK} evaluator=eval_hpt_standard \
    evaluator.init_mode=seeds "+evaluator.init_seeds=${SEEDS}" \
    "+evaluator.rollout_mode=chunk_openloop" "+evaluator.chunk_k=8" "+evaluator.temporal_ensemble=false" \
    "++evaluator.env_kwargs.obstacle_level=${LVL}" evaluator.max_steps=1800 "++evaluator.video_fps=30" "++evaluator.max_videos=1" \
    evaluator.coverage_threshold=0.95 evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
    trainer=debug trainer.profiler=null logger=csv \
    data.train_datasets.pushshapes_sim.resolver.folder_path=${DATASET} \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=${DATASET} \
    data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
    data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal_eval \
    data.train_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
    data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
    norm_stats.norm_mode=quantile norm_stats.precomputed_norm_path=${NORM} ${EXTRA}
}
run_packed() {
  srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    name=g2_${CELL}_L${LVL} mode=eval data=tsimulation model=${MODEL} \
    ckpt_path=${CK} evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
    evaluator.init_mode=seeds "+evaluator.init_seeds=${SEEDS}" \
    "++evaluator.env_kwargs.obstacle_level=${LVL}" evaluator.max_steps=1800 "++evaluator.video_fps=30" "++evaluator.max_videos=1" \
    evaluator.coverage_threshold=0.95 evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
    trainer=debug trainer.profiler=null logger=csv \
    data.train_datasets.pushshapes_sim.resolver.folder_path=${DATASET} \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=${DATASET} \
    data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_eval \
    data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_eval \
    norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=${NORM} ${EXTRA}
}
ok=0
for attempt in 1 2 3; do
  if [ "$FAMILY" = "hpt" ]; then run_hpt; else run_packed; fi
  rc=$?
  if [ "$rc" -eq 0 ]; then ok=1; break; fi
  echo "[retry] attempt $attempt rc=$rc (likely CUDA-busy/transient); sleep 45"; sleep 45
done
echo "GENLVL_${CELL}_L${LVL}_OK=${ok}"
