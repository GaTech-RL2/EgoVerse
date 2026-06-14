#!/bin/bash
#SBATCH --job-name=tfOverlayAct8
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/tfoverlay_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/tfoverlay_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
L=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs
DSP=/coc/flash7/paphiwetsa3/datasets

# Teacher-forced path overlay (predicted k=8 chunk red vs GT green, per-frame anim).
# act8 -> K=8 override (script default K = trunk action_horizon=32, wrong for act8 head).

# c200 act8 (pushshapes_paper, fixed-goal)
RUN_DIR=$L/hptFlowC200CausalAct8/hpt_flow_pushshapes_paper_causal_obs1_act8_2026-06-14_07-56-50 \
CKPT=epoch_epoch=1099.ckpt DATA=$DSP/pushshapes_paper \
OUTDIR=$L/tf_overlay_act8/c200act8 LABEL=c200act8 EPISODES=0,2,4 OUTPUT_MODE=anim K=8 FPS=12 \
python scripts/tf_path_hpt_flow.py
echo "C200_EXIT=$?"

# c3000 act8 (circle_3000)
RUN_DIR=$L/hptFlowC3000CausalAct8/hpt_flow_circle_3000_causal_obs1_act8_2026-06-14_07-56-46 \
CKPT=epoch_epoch=899.ckpt DATA=$DSP/circle_3000 \
OUTDIR=$L/tf_overlay_act8/c3000act8 LABEL=c3000act8 EPISODES=0,2,4 OUTPUT_MODE=anim K=8 FPS=12 \
python scripts/tf_path_hpt_flow.py
echo "C3000_EXIT=$?"
