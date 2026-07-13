#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=01:15:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --exclude=ig-88,spd-13,sonny,cyborg,hk47,heistotron,baymax,deebot,kitt,cheetah,megazord,puma
#SBATCH --output=bf_chunkviz_%j.out
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
PY=.venv/bin/python
mkdir -p chunkviz_bf eval_bf

# --- snap cvae_zs_var4 last.ckpt with its TRUE epoch (provenance rule) ---
RUN_C=$(ls -dt logs/indomain_c4/bf_cvae_zs_var4_2026-* | head -1)
cp -p "$RUN_C/checkpoints/last.ckpt" eval_bf/bf_cvae_zs_var4_new_snap.ckpt
E=$($PY -c "import torch;print(int(torch.load('eval_bf/bf_cvae_zs_var4_new_snap.ckpt',map_location='cpu')['epoch']))")
echo "CVAE_ZS_VAR4_EPOCH=$E"
mv eval_bf/bf_cvae_zs_var4_new_snap.ckpt "eval_bf/bf_cvae_zs_var4_ep${E}_snap.ckpt"

# --- snap prdec_var4 ep1499 (documents the ep999->1499 seam collapse) ---
RUN_P=logs/indomain_c4/bf_prdec_var4_2026-07-10_15-23-17
[ -f eval_bf/bf_prdec_var4_ep1499_snap.ckpt ] || \
  cp -p "$RUN_P/checkpoints/epoch_epoch=1499.ckpt" eval_bf/bf_prdec_var4_ep1499_snap.ckpt

$PY -m egomimic.eval.explorer.export --ckpt "eval_bf/bf_cvae_zs_var4_ep${E}_snap.ckpt" \
  --config-path "$RUN_C/.hydra/config.yaml" \
  --out "chunkviz_bf/cvae_zs_var4_ep${E}.npz" --n-episodes 10
echo "EXPORT_cvae_rc=$?"
$PY -m egomimic.eval.explorer.export --ckpt eval_bf/bf_prdec_var4_ep999_snap.ckpt \
  --config-path "$RUN_P/.hydra/config.yaml" \
  --out chunkviz_bf/prdec_var4_ep999v2.npz --n-episodes 10
echo "EXPORT_p999_rc=$?"
$PY -m egomimic.eval.explorer.export --ckpt eval_bf/bf_prdec_var4_ep1499_snap.ckpt \
  --config-path "$RUN_P/.hydra/config.yaml" \
  --out chunkviz_bf/prdec_var4_ep1499.npz --n-episodes 10
echo "EXPORT_p1499_rc=$?"

$PY -m egomimic.eval.explorer.build_html \
  --model "cvae_zs_var4@ep${E}=chunkviz_bf/cvae_zs_var4_ep${E}.npz" \
  --model "prdec_var4@ep999=chunkviz_bf/prdec_var4_ep999v2.npz" \
  --model "prdec_var4@ep1499=chunkviz_bf/prdec_var4_ep1499.npz" \
  --out chunkviz_bf/var4_family.html
echo "BUILD_rc=$?"
echo BF_CHUNKVIZ_DONE
