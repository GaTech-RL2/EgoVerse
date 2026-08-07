#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --exclude=ig-88,spd-13,sonny,cyborg,hk47,heistotron,baymax,deebot,kitt,cheetah,megazord,puma,omgwth,megabot
#SBATCH --output=bf_chunkviz_%j.out
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
PY=.venv/bin/python
RUN_P=logs/indomain_c4/bf_prdec_var4_2026-07-10_15-23-17
[ -f eval_bf/bf_prdec_var4_ep499_snap.ckpt ] || \
  cp -p "$RUN_P/checkpoints/epoch_epoch=499.ckpt" eval_bf/bf_prdec_var4_ep499_snap.ckpt
$PY -m egomimic.eval.explorer.export --ckpt eval_bf/bf_prdec_var4_ep499_snap.ckpt \
  --config-path "$RUN_P/.hydra/config.yaml" \
  --out chunkviz_bf/prdec_var4_ep499.npz --n-episodes 10
echo "EXPORT_rc=$?"
$PY -m egomimic.eval.explorer.build_html \
  --model "prdec_var4@ep499=chunkviz_bf/prdec_var4_ep499.npz" \
  --model "prdec_var4@ep999=chunkviz_bf/prdec_var4_ep999v2.npz" \
  --model "prdec_var4@ep1499=chunkviz_bf/prdec_var4_ep1499.npz" \
  --out chunkviz_bf/prdec_var4_evolution.html
echo "BUILD_rc=$?"
echo BF_CHUNKVIZ_DONE
