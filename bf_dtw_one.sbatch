#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --output=bf_dtw_one_%j.out
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
A=bf_prdec_sig
D=$(ls -dt logs/indomain_c4/${A}_2026-07-10_1[4-9]* | head -1)
CK=$D/checkpoints/last.ckpt
CFG=eval_bf/dtw/${A}_dtwcfg.yaml
.venv/bin/python scripts/bf_dtw_latent.py --ckpt "$CK" --config-path "$CFG" --n-episodes 6 --evr 0.95 --good-eps 3,7,0
echo ONE_DONE rc=$?
