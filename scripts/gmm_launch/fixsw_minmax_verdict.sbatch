#!/bin/bash
#SBATCH --job-name=fixsw_minmax_verdict
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/fixsw_minmax_verdict_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## minmax_e15p ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/minmax_e15p_ep1499
echo "COV_minmax_e15p_ep1499_EXIT=$?"
echo "########## minmax_replay ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir fixsweep/minmax_replay_ep1499
echo "COV_minmax_replay_ep1499_EXIT=$?"
echo "DONE_minmax_verdict"
