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
$PY -m egomimic.eval.explorer.export --ckpt eval_bf/bf_prdec_nopre_gmm_ep999_snap.ckpt --config-path logs/indomain_c4/bf_prdec_nopre_gmm_2026-07-10_23-26-53/.hydra/config.yaml --out chunkviz_bf/nopre_gmm_ep999.npz --n-episodes 10
echo EXPORT1_rc=$?
$PY -m egomimic.eval.explorer.export --ckpt eval_bf/bf_prdec_nopre_gmm_ep2999_snap.ckpt --config-path logs/indomain_c4/bf_prdec_nopre_gmm_2026-07-10_23-26-53/.hydra/config.yaml --out chunkviz_bf/nopre_gmm_ep2999.npz --n-episodes 10
echo EXPORT2_rc=$?
$PY -m egomimic.eval.explorer.build_html --model "nopre_gmm@ep999=chunkviz_bf/nopre_gmm_ep999.npz" --model "nopre_gmm@ep2999=chunkviz_bf/nopre_gmm_ep2999.npz" --out chunkviz_bf/nopre_gmm_organization.html
echo BUILD_rc=$?
echo DONE
