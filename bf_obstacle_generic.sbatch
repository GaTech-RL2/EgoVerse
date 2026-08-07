#!/bin/bash
#SBATCH --job-name=bfObsGen
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --exclude=ig-88,spd-13,sonny,cyborg,hk47,heistotron,baymax,deebot,kitt,cheetah,megazord,puma,omgwth,megabot
#SBATCH --output=bf_obsgen_%A_L%a.out
# args: TAG CK CFG. task = obstacle level; both embs per task; skip-guard.
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
PY=.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=egl
TAG=$1; CK=$2; CFG=$3; L=${SLURM_ARRAY_TASK_ID}
mkdir -p eval_bf/obst_${TAG}
for SPEC in "15:circle" "17:circle_small"; do
  EMB=${SPEC%%:*}; PU=${SPEC##*:}
  OUT=eval_bf/obst_${TAG}/L${L}_e${EMB}
  [ -s "${OUT}.log" ] && grep -q ep_coverages "${OUT}.log" && { echo "L$L e$EMB done"; continue; }
  $PY -m egomimic.eval.core.ckpt_loading --ckpt "$CK" --config-path "$CFG"     --eval-class packed --n-episodes 5 --max-steps 1800 --obs-stride 4     --coverage-threshold 1.01 --full-horizon --max-coverage     --embodiment-name pushshapes_sim --only-emb "$EMB" --pusher "$PU"     --init-mode seeds --init-seed-base 0 --rng-pairing     --obstacle-level "$L" --out-dir "$OUT" > "${OUT}.log" 2>&1
  echo "L$L e$EMB rc=$?"
done
echo TASK_DONE
