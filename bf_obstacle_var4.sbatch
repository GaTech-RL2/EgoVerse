#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --exclude=ig-88,spd-13,sonny,cyborg,hk47,heistotron,baymax,deebot,kitt,cheetah,megazord,puma,omgwth,megabot
#SBATCH --output=bf_obst_var4_%A_%a.out
# 30-level obstacle sweep on bf_prdec_var4@ep999: task = level, 5 seeds [0..4], both embs.
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
PY=.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
L=${SLURM_ARRAY_TASK_ID}
CK=eval_bf/bf_prdec_var4_ep999_snap.ckpt
CFG=logs/indomain_c4/bf_prdec_var4_2026-07-10_15-23-17/.hydra/config.yaml
mkdir -p eval_bf/obst_var4
for SPEC in "15:circle" "17:circle_small"; do
  EMB=${SPEC%%:*}; PU=${SPEC##*:}
  OUT=eval_bf/obst_var4/L${L}_e${EMB}
  $PY -m egomimic.eval.core.ckpt_loading --ckpt "$CK" --config-path "$CFG" \
    --eval-class packed --n-episodes 5 --max-steps 1800 --obs-stride 4 \
    --coverage-threshold 1.01 --full-horizon --max-coverage \
    --embodiment-name pushshapes_sim --only-emb "$EMB" --pusher "$PU" \
    --init-mode seeds --init-seed-base 0 --rng-pairing \
    --obstacle-level "$L" \
    --out-dir "$OUT" > "${OUT}.log" 2>&1
  echo "OBST_L${L}_e${EMB}_rc=$?"
done
echo OBST_TASK_DONE
