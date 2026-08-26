#!/bin/bash
#SBATCH --job-name=smallgen
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=ig-88
#SBATCH --requeue
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/smallgen/%x_%A_%a.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/smallgen/%x_%A_%a.err
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
# Zero-shot: big-circle +gen HPT model rolled out on SMALL-circle pusher, obstacle levels.
RUN=hptFlowC3000Act8PlusGen
DATASET=/coc/flash7/paphiwetsa3/datasets/circle_3000_plus_cotrain240
LVL=${SLURM_ARRAY_TASK_ID:-0}
RD=$(ls -dt logs/${RUN}/*/ 2>/dev/null | head -1)
CK=${RD}checkpoints/last.ckpt
NORM=${RD}norm_stats/norm_stats.json
VID=logs/smallgen_vids/lvl${LVL}; mkdir -p "$VID"; export ROLLOUT_VIDEO_DIR=$VID
echo "SMALLGEN RUN=$RUN LVL=$LVL pusher=circle_small seed=$LVL CK=$CK VID=$VID"
ok=0
for attempt in 1 2 3; do
  srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    name=sg_L${LVL} mode=eval data=tsimulation model=hpt_pushshapes_circle \
    ++model.robomimic_model.head_specs.pushshapes_sim.action_horizon=8 \
    ++model.robomimic_model.head_specs.pushshapes_sim.model.act_seq=8 \
    ckpt_path=${CK} evaluator=eval_hpt_standard \
    evaluator.init_mode=seeds "+evaluator.init_seeds=[${LVL}]" \
    "+evaluator.rollout_mode=chunk_openloop" "+evaluator.chunk_k=8" "+evaluator.temporal_ensemble=false" \
    "++evaluator.env_kwargs.obstacle_level=${LVL}" "++evaluator.env_kwargs.pusher_shape=circle_small" \
    evaluator.max_steps=1800 "++evaluator.video_fps=30" "++evaluator.max_videos=1" \
    evaluator.coverage_threshold=0.95 evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
    trainer=debug trainer.profiler=null logger=csv \
    data.train_datasets.pushshapes_sim.resolver.folder_path=${DATASET} \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=${DATASET} \
    data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
    data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal_eval \
    data.train_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
    data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
    norm_stats.norm_mode=quantile norm_stats.precomputed_norm_path=${NORM}
  rc=$?
  if [ "$rc" -eq 0 ]; then ok=1; break; fi
  echo "[retry] attempt $attempt rc=$rc; sleep 45"; sleep 45
done
echo "SMALLGEN_L${LVL}_OK=${ok}"
