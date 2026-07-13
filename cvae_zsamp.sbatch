#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --exclude=ig-88,spd-13,sonny,cyborg,hk47,heistotron,baymax,deebot,kitt,cheetah,megazord,puma
#SBATCH --output=cvae_tf_%j.out
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
CK=$(ls logs/indomain_c4/bf_cvae_zs_bcrecipe_2026-07-10_1[4-9]*/checkpoints/epoch_epoch=499.ckpt | head -1)
echo "[overlay cvae_zs_ep499] ckpt=$CK"
.venv/bin/python scripts/bf_z_samples.py --ckpt "$CK" --k 8 --eps-per-emb 2 --out-dir bf_viz --tag cvae_zs_ep499_zsamp
echo TF_DONE rc=$?
