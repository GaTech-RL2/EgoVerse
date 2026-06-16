#!/bin/bash
#SBATCH --job-name=fixsw_fairmatchw
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/fixsw_fairmatchw_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## fairmatch_sb2000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 2000 --rng-pairing --out-dir fixsweep/fairmatch_sb2000_ep1499
echo "COV_fairmatch_sb2000_ep1499_EXIT=$?"
echo "########## fairmatch_sb3000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 3000 --rng-pairing --out-dir fixsweep/fairmatch_sb3000_ep1499
echo "COV_fairmatch_sb3000_ep1499_EXIT=$?"
echo DONE_fairmatchw
