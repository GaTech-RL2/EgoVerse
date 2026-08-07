#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=00:50:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --output=vid_probe_%j.out
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
PY=.venv/bin/python
NAME=$1; CK=$2
OUT=eval_bf/vid_${NAME}
$PY -m egomimic.eval.core.ckpt_loading --ckpt "$CK" --config-path eval_bf_prdec40_cfg.yaml   --eval-class packed --n-episodes 3 --max-steps 1800 --obs-stride 4   --coverage-threshold 1.01 --full-horizon --max-coverage --per-episode-videos   --embodiment-name pushshapes_sim --only-emb 15 --pusher circle   --init-mode seeds --init-seed-base 0 --rng-pairing   --out-dir "$OUT" > "${OUT}.log" 2>&1
echo VID_${NAME}_rc=$?
