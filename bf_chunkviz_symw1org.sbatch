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
D=logs/indomain_c4/bf_symw1_gmm_2026-07-11_23-06-12
$PY -m egomimic.eval.explorer.export --ckpt eval_bf/bf_symw1_gmm_ep999_snap.ckpt --config-path logs/indomain_c4/bf_symw1_gmm_2026-07-11_23-06-12/.hydra/config.yaml --out chunkviz_bf/symw1_gmm_ep999.npz --n-episodes 10
echo EXPORT1_rc=$?
$PY -m egomimic.eval.explorer.export --ckpt eval_bf/bf_symw1_gmm_ep1999_snap.ckpt --config-path logs/indomain_c4/bf_symw1_gmm_2026-07-11_23-06-12/.hydra/config.yaml --out chunkviz_bf/symw1_gmm_ep1999.npz --n-episodes 10
echo EXPORT2_rc=$?
$PY -m egomimic.eval.explorer.build_html --model "symw1_gmm@ep999=chunkviz_bf/symw1_gmm_ep999.npz" --model "symw1_gmm@ep1999=chunkviz_bf/symw1_gmm_ep1999.npz" --out chunkviz_bf/symw1_gmm_organization.html
echo BUILD_rc=$?
echo DONE
