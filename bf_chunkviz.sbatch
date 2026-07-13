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
for SPEC in "bf_cvae_zs_do1:cvae_zs_do1" "bf_cvae_zs_var4:cvae_zs_var4"; do
  A=${SPEC%%:*}; TAG=${SPEC##*:}
  D=$(ls -dt logs/indomain_c4/${A}_2026-07-1* | head -1)
  CK=eval_bf/${A}_ep499_snap.ckpt
  $PY -m egomimic.eval.explorer.export --ckpt "$CK" --config-path "$D/.hydra/config.yaml" \
    --out chunkviz_bf/${TAG}_ep499.npz --n-episodes 10
  echo "EXPORT_${TAG}_rc=$?"
done
$PY -m egomimic.eval.explorer.build_html \
  --model "cvae_zs_do1@ep499=chunkviz_bf/cvae_zs_do1_ep499.npz" \
  --model "cvae_zs_var4@ep499=chunkviz_bf/cvae_zs_var4_ep499.npz" \
  --out chunkviz_bf/cvae_fixarms_ep499.html
echo BUILD_rc=$?
echo BF_CHUNKVIZ_DONE
