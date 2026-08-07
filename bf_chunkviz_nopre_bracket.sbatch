#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=01:00:00
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
RUN=logs/indomain_c4/bf_prdec_nopre_gmm_2026-07-10_23-26-53
for EP in 1499 1999 2499; do
  SNAP=eval_bf/bf_prdec_nopre_gmm_ep${EP}_snap.ckpt
  [ -s "$SNAP" ] || cp "$RUN/checkpoints/epoch_epoch=${EP}.ckpt" "$SNAP"
  TRUE=$($PY -c "import torch;print(torch.load('$SNAP',map_location='cpu',weights_only=False)['epoch'])")
  echo "SNAP ep${EP} true_epoch=${TRUE}"
  $PY -m egomimic.eval.explorer.export --ckpt "$SNAP" \
    --config-path $RUN/.hydra/config.yaml \
    --out chunkviz_bf/nopre_gmm_ep${EP}.npz --n-episodes 10
  echo EXPORT_ep${EP}_rc=$?
done
$PY chunklen_report.py chunkviz_bf/nopre_gmm_ep1499.npz chunkviz_bf/nopre_gmm_ep1999.npz chunkviz_bf/nopre_gmm_ep2499.npz
echo DONE
