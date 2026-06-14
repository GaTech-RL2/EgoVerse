#!/bin/bash
#SBATCH --job-name=tfpathVid
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/tfpathVid_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/tfpathVid_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OUTPUT_MODE=video FPS=2 EPISODES=0,1,2,3,4

# ---- Run A: hptFlowC3000Causal (ep1799), dataset circle_3000 ----
export RUN_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC3000Causal/hpt_flow_circle_3000_causal_obs1_act32_2026-06-13_03-08-51
export CKPT='epoch_epoch=1799.ckpt'
export DATA=/coc/flash7/paphiwetsa3/datasets/circle_3000
export LABEL=c3000
export OUTDIR=${RUN_DIR}/eval/tfpath_vid
srun python scripts/tf_path_hpt_flow.py
echo "RUN_A_EXIT=$?"

# ---- Run B: hptFlowC200Causal (last.ckpt), dataset pushshapes_paper ----
export RUN_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC200Causal_pushshapesPaper/hpt_flow_pushshapes_paper_causal_obs1_act32_2026-06-13_06-03-25
export CKPT='last.ckpt'
export DATA=/coc/flash7/paphiwetsa3/datasets/pushshapes_paper
export LABEL=c200causal
export OUTDIR=${RUN_DIR}/eval/tfpath_vid
srun python scripts/tf_path_hpt_flow.py
echo "RUN_B_EXIT=$?"
echo "ALL_DONE"
