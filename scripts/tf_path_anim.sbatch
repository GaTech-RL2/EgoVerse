#!/bin/bash
#SBATCH --job-name=tfpath_anim
#SBATCH -A overcap
#SBATCH --partition=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:l40s:1
#SBATCH -c4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH -o /coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/tf_path_out/anim_%j.log

source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2

export SANITY=0
export OUTPUT_MODE=anim
export FPS=12
export EPISODES=0,1,2,3,4

echo "============ RUN A: c3000 causal (anim) ============"
export RUN_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC3000Causal/hpt_flow_circle_3000_causal_obs1_act32_2026-06-13_03-08-51
export CKPT=epoch_epoch=1799.ckpt
export DATA=/coc/flash7/paphiwetsa3/datasets/circle_3000
export OUTDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/tf_path_out/c3000_tfpath_anim
export LABEL=c3000
python scripts/tf_path_hpt_flow.py
echo "RUN_A_EXIT=$?"

echo "============ RUN B: c200 causal pushshapes_paper (anim) ============"
export RUN_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC200Causal_pushshapesPaper/hpt_flow_pushshapes_paper_causal_obs1_act32_2026-06-13_06-03-25
export CKPT=last.ckpt
export DATA=/coc/flash7/paphiwetsa3/datasets/pushshapes_paper
export OUTDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/tf_path_out/c200causal_tfpath_anim
export LABEL=c200causal
python scripts/tf_path_hpt_flow.py
echo "RUN_B_EXIT=$?"

echo "ALL_DONE"
