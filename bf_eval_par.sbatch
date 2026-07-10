#!/bin/bash
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --qos=short
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=ig-88,voltron,sonny,cyborg,hk47,spd-13,crushinator,heistotron,baymax,deebot,kitt,cheetah
#SBATCH --requeue
# BATCHFLOW parallel 40-seed eval. args: NAME CK CFG. tasks 0-3=emb15 circle, 4-7=emb17 small.
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
PY=.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
NAME=$1; CK=$2; CFG=$3
T=${SLURM_ARRAY_TASK_ID}
if [ "$T" -lt 4 ]; then EMB=15; PU=circle;       BASE=$(( T*10 ));
else                    EMB=17; PU=circle_small; BASE=$(( (T-4)*10 )); fi
mkdir -p eval_bf
OUT=eval_bf/${NAME}_par_e${EMB}_s${BASE}
$PY -m egomimic.eval.core.ckpt_loading --ckpt "$CK" --config-path "$CFG" \
  --eval-class packed --n-episodes 10 --max-steps 1800 --obs-stride 4 \
  --coverage-threshold 1.01 --full-horizon --max-coverage \
  --embodiment-name pushshapes_sim --only-emb "$EMB" --pusher "$PU" \
  --init-mode seeds --init-seed-base "$BASE" --rng-pairing \
  --out-dir "$OUT" > "${OUT}.log" 2>&1
echo "BF_PAR_${NAME}_e${EMB}_s${BASE}_rc=$?"
