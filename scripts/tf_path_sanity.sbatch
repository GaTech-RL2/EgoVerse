#!/bin/bash
#SBATCH --job-name=tfpath_sanity
#SBATCH -A overcap
#SBATCH --partition=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:l40s:1
#SBATCH -c4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH -o /coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/tf_path_out/sanity_%j.log

source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2

export SANITY=1
export RUN_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC200Causal_pushshapesPaper/hpt_flow_pushshapes_paper_causal_obs1_act32_2026-06-13_06-03-25
export CKPT=last.ckpt
export DATA=/coc/flash7/paphiwetsa3/datasets/pushshapes_paper
export OUTDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/tf_path_out/sanity
export LABEL=c200causal
export EPISODES=0

python scripts/tf_path_hpt_flow.py
echo "SANITY_DONE_EXIT=$?"
