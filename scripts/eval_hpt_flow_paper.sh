#!/bin/bash
#SBATCH --job-name=evalHptFlowPaper
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalHptFlowPaper_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalHptFlowPaper_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200

RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowPaper/hpt_flow_pushshapes_paper_2026-06-11_05-07-03
CKPT=${RUNDIR}/checkpoints/eval_snapshot_paper.ckpt
NORM=${RUNDIR}/norm_stats/norm_stats.json
DS=/coc/flash7/paphiwetsa3/datasets/pushshapes_paper

# Video rendering: SimRolloutEval writes per-episode mp4s when ROLLOUT_VIDEO_DIR is set.
export ROLLOUT_VIDEO_DIR=${RUNDIR}/eval/video_paper_replay_ep299
mkdir -p "$ROLLOUT_VIDEO_DIR"

# FIXED-GOAL dataset: init_mode=replay reproduces each recorded episode init state +
# its fixed goal [256,256,0.79]. limit_val_batches=4 -> replay 4 recorded episodes.
# coverage_threshold left at the eval_hpt_standard wired default (0.7).
# norm_mode=quantile to match how the run was trained.
srun python -m egomimic.trainHydra   --config-name=train_zarr_cartesian   name=evalHptFlowPaper description=eval_hpt_flow_paper_replay_ep299 mode=eval   data=tsimulation model=hpt_pushshapes_circle evaluator=eval_hpt_standard   ckpt_path=${CKPT}   evaluator.init_mode=replay   evaluator.max_steps=400   evaluator.limit_val_batches=4 trainer.limit_val_batches=4   trainer=debug trainer.profiler=null logger=csv   data.train_datasets.pushshapes_sim.resolver.folder_path=$DS   data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS   norm_stats.norm_mode=quantile   norm_stats.precomputed_norm_path=$NORM
echo "EVAL_EXIT=$?"
