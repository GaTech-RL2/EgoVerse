#!/bin/bash
#SBATCH --job-name=dsCotEval
#SBATCH --account=hoffman-lab
#SBATCH --partition=hoffman-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-dualstream/auto_eval_logs/%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-dualstream/auto_eval_logs/%x_%j.out
#
# Cotrain closed-loop coverage eval for a DUAL-STREAM run (BOTH embs), arch-agnostic:
# the model is rebuilt from the run's OWN .hydra/config.yaml, so the dual_stream
# _targets_ resolve automatically -- REQUIRES PYTHONPATH=. so they hit the CLONE's
# egomimic (with the dual-stream code), not a site-packages egomimic.
#
# usage: sbatch dualstream_coteval.sh <run_glob_under_logs> <TAG>
#   asym: sbatch dualstream_coteval.sh indomain_c4/dualstream_cotrain_2026     ds_asym
#   sym : sbatch dualstream_coteval.sh indomain_c4/dualstream_cotrain_sym_2026 ds_sym
# env knobs (all optional):
#   SEEDS_PER_LEVEL (5)  obstacle-sweep seeds per level
#   THRESH (1.01)        coverage_threshold; >=1.0 => no early-stop => unbiased true peak
#   MOTION (1)           also run the obstacle-free 40-seed in-domain check
#   LEVELS ("0 .. 29")   obstacle levels to sweep (override for a fast smoke, e.g. LEVELS=0)
#   OBS_STRIDE (auto)    AR commit stride; default = head chunk_len from the config (fallback 4)
#   CKPT (newest last)   explicit checkpoint path to eval a fixed epoch milestone
set -uo pipefail
G=/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-dualstream
cd "$G"; export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
PY=.venv/bin/python
GLOB=${1:?usage: sbatch dualstream_coteval.sh <run_glob_under_logs> <TAG>}
TAG=${2:?need TAG}
SPL=${SEEDS_PER_LEVEL:-5}
THRESH=${THRESH:-1.01}
MOTION=${MOTION:-1}
LEVELS=${LEVELS:-$(seq 0 29)}
mkdir -p auto_eval_logs fixsweep

RD=$(ls -dt logs/${GLOB}*/ 2>/dev/null | grep -v SMOKE | head -1)
[ -n "$RD" ] || { echo "NO RUN DIR for glob logs/${GLOB}*"; exit 1; }
SRCK=${CKPT:-$(ls -t ${RD}checkpoints/last.ckpt 2>/dev/null | head -1)}
[ -f "$SRCK" ] || { echo "NO CKPT (looked at ${RD}checkpoints/last.ckpt; set CKPT= to override)"; exit 1; }
# snapshot the ckpt to a stable path so a live training run can't overwrite it mid-eval
CK="${RD}checkpoints/coteval_snap_${TAG}.ckpt"; cp "$SRCK" "$CK"
CFG="${RD}.hydra/config.yaml"; CFGBS="${RD}.hydra/config_eval_bs.yaml"
# val batch_size -> 40 so episodes-run = min(bs, n-episodes) is never the cap (both cotrain keys)
$PY -c "
from omegaconf import OmegaConf
c=OmegaConf.load('$CFG'); v=c.data.valid_dataloader_params
for k in ('pushshapes_sim','pushshapes_sim_small_circle'):
    if k in v: v[k].batch_size=40
OmegaConf.save(c,'$CFGBS'); print('patched bs ->', [k for k in v])
"
# obs_stride = the run's output-head chunk_len unless overridden (NoChunk head chunk_len=4)
CL=${OBS_STRIDE:-$(grep -m1 -E "chunk_len:" "$CFG" 2>/dev/null | awk '{print $2}')}
[ -z "$CL" ] && CL=4
echo "DS_COTEVAL tag=$TAG RD=$RD SRCK=$SRCK CK=$CK spl=$SPL thresh=$THRESH motion=$MOTION obs_stride=$CL"

for pair in "15 circle" "17 circle_small"; do
  set -- $pair; EMB=$1; PU=$2
  if [ "$MOTION" = "1" ]; then
    for BASE in 0 20; do
      echo "##### $TAG MOTION emb$EMB $PU obs0 seeds ${BASE}..$((BASE+19)) #####"
      $PY -m egomimic.eval.core.ckpt_loading --ckpt "$CK" --config-path "$CFGBS" --eval-class packed \
        --n-episodes 20 --max-steps 1800 --obs-stride $CL --max-coverage --coverage-threshold $THRESH \
        --only-emb $EMB --pusher $PU --obstacle-level 0 \
        --init-mode seeds --init-seed-base $BASE --rng-pairing \
        --out-dir fixsweep/${TAG}_motion_e${EMB}_s${BASE} 2>&1
      echo "${TAG}_motion_e${EMB}_s${BASE}_rc=$?"
    done
  fi
  for L in $LEVELS; do
    echo "##### $TAG OBJGEN emb$EMB $PU obstacle_level=$L n=$SPL #####"
    $PY -m egomimic.eval.core.ckpt_loading --ckpt "$CK" --config-path "$CFGBS" --eval-class packed \
      --n-episodes $SPL --max-steps 1800 --obs-stride $CL --max-coverage --coverage-threshold $THRESH \
      --only-emb $EMB --pusher $PU --obstacle-level $L \
      --init-mode seeds --init-seed-base 0 --rng-pairing \
      --out-dir fixsweep/${TAG}_obj_e${EMB}_L${L} 2>&1
    echo "${TAG}_obj_e${EMB}_L${L}_rc=$?"
  done
done
echo "DS_COTEVAL_DONE $TAG"
