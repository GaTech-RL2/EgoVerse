#!/bin/bash
#SBATCH --job-name=bf_overlay
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --exclude=kitt,cyborg,ig-88,hk47,spd-13,sonny,heistotron,megazord,puma,baymax,deebot,cheetah
#SBATCH --output=bf_overlay_%j.out
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
TAG=$1
CK=$(ls -t logs/debug/bf_overfit_${TAG}_*/checkpoints/last.ckpt | head -1)
echo "[overlay $TAG] ckpt=$CK"
srun .venv/bin/python scripts/bf_overlay.py --ckpt "$CK" --out-dir bf_viz --tag "$TAG"
