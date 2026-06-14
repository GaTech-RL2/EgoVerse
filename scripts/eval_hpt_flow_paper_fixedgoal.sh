#!/bin/bash
#SBATCH --job-name=evalHptFlowPaperFG
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalHptFlowPaperFG_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalHptFlowPaperFG_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200

RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowPaper/hpt_flow_pushshapes_paper_2026-06-11_05-07-03
CKPT=${RUNDIR}/checkpoints/eval_snapshot_paper.ckpt
NORM=${RUNDIR}/norm_stats/norm_stats.json
DS=/coc/flash7/paphiwetsa3/datasets/pushshapes_paper

# Per-episode mp4 dump dir.
export ROLLOUT_VIDEO_DIR=${RUNDIR}/eval/video_paper_fixedgoal_randinit_ep299
mkdir -p "$ROLLOUT_VIDEO_DIR"

# FIXED-GOAL + RANDOM-INIT-T eval (the correct eval for a fixed-goal dataset):
#   init_mode=seeds + init_seeds=[0..19] -> each seed rejection-samples a RANDOM
#   object(T)/goal/pusher; fixed_goal=[256,256,pi/4] then overrides ONLY the goal
#   (set_state with agent/object None) so T+pusher stay random while goal is fixed
#   at the training target.  theta = pi/4 = 0.7853981633974483 (read from data:
#   all 200 pushshapes_paper episodes have goal_pose row0 = [256,256,0.7853981634]).
# data=tsimulation = PACKED loader (per-frame loader silently rolls out 0 episodes).
# coverage_threshold left at the eval_hpt_standard wired default (0.7).
srun python -m egomimic.trainHydra   --config-name=train_zarr_cartesian   name=evalHptFlowPaperFG description=eval_hpt_flow_paper_fixedgoal_randinit_ep299 mode=eval   data=tsimulation model=hpt_pushshapes_circle evaluator=eval_hpt_standard   ckpt_path=${CKPT}   evaluator.init_mode=seeds   '+evaluator.init_seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]'   '++evaluator.fixed_goal=[256.0,256.0,0.7853981633974483]'   evaluator.max_steps=400   evaluator.limit_val_batches=1 trainer.limit_val_batches=1   trainer=debug trainer.profiler=null logger=csv   data.train_datasets.pushshapes_sim.resolver.folder_path=$DS   data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS   norm_stats.norm_mode=quantile   norm_stats.precomputed_norm_path=$NORM
echo "EVAL_EXIT=$?"
