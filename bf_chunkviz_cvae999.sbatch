#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --exclude=ig-88,spd-13,sonny,cyborg,hk47,heistotron,baymax,deebot,kitt,cheetah,megazord,puma,omgwth
#SBATCH --output=bf_chunkviz_%j.out
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
PY=.venv/bin/python
RUN_C=logs/indomain_c4/bf_cvae_zs_var4_2026-07-11_23-49-38
$PY -m egomimic.eval.explorer.export --ckpt eval_bf/bf_cvae_zs_var4_ep999_snap.ckpt \
  --config-path "$RUN_C/.hydra/config.yaml" \
  --out chunkviz_bf/cvae_zs_var4_ep999.npz --n-episodes 10
echo "EXPORT_rc=$?"
$PY -m egomimic.eval.explorer.build_html \
  --model "cvae_zs_var4@ep999=chunkviz_bf/cvae_zs_var4_ep999.npz" \
  --model "cvae_zs_var4@ep499=chunkviz_bf/cvae_zs_var4_ep499.npz" \
  --model "prdec_var4@ep999=chunkviz_bf/prdec_var4_ep999v2.npz" \
  --model "prdec_var4@ep1499=chunkviz_bf/prdec_var4_ep1499.npz" \
  --out chunkviz_bf/var4_family.html
echo "BUILD_rc=$?"
echo BF_CHUNKVIZ_DONE
