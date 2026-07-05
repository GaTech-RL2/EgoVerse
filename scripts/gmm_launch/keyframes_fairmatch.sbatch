#!/bin/bash
#SBATCH --job-name=kf_fairmatch
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/kf_fairmatch_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
echo "########## keyframes fairmatch ep1499 ##########"
.venv/bin/python scripts/keyframe_projection.py --ckpt "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/.hydra/config.yaml" --n-episodes 6 --out-dir keyframes_out/fairmatch_ep1499
echo "KF_EXIT=$?"
