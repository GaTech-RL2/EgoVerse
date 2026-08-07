#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=00:45:00
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
mkdir -p chunkviz_bf
for SPEC in "bf_prdec_var4:prdec_var4"; do
  A=${SPEC%%:*}; TAG=${SPEC##*:}
  D=$(ls -dt logs/indomain_c4/${A}_2026-07-1* | head -1)
  CK=eval_bf/${A}_ep999_snap.ckpt
  $PY -m egomimic.eval.explorer.export --ckpt "$CK" --config-path "$D/.hydra/config.yaml" \
    --out chunkviz_bf/${TAG}_ep999.npz --n-episodes 10
  echo "EXPORT_${TAG}_rc=$?"
done
$PY -m egomimic.eval.explorer.build_html \
  --model "prdec_var4@ep999=chunkviz_bf/prdec_var4_ep999.npz" --out chunkviz_bf/prdec_var4_ep999.html
echo BUILD_rc=$?
echo BF_CHUNKVIZ_DONE
