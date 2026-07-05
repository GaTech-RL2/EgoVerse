#!/bin/bash
#SBATCH --job-name=vj_rsched
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=embers
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/vj_rsched_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## rsched_seeds ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_s1c1_rsched/gmm_circle_s1c1_rsched_200k_2026-06-10_17-40-39/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_s1c1_rsched/gmm_circle_s1c1_rsched_200k_2026-06-10_17-40-39/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir vj/rsched_seeds_ep1499
echo "COV_rsched_seeds_ep1499_EXIT=$?"
echo "########## rsched_replay ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_s1c1_rsched/gmm_circle_s1c1_rsched_200k_2026-06-10_17-40-39/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_s1c1_rsched/gmm_circle_s1c1_rsched_200k_2026-06-10_17-40-39/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle  --out-dir vj/rsched_replay_ep1499
echo "COV_rsched_replay_ep1499_EXIT=$?"
echo "########## rsched_seeds ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_s1c1_rsched/gmm_circle_s1c1_rsched_200k_2026-06-10_17-40-39/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_circle_s1c1_rsched/gmm_circle_s1c1_rsched_200k_2026-06-10_17-40-39/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir vj/rsched_seeds_ep1999
echo "COV_rsched_seeds_ep1999_EXIT=$?"
echo "########## rsched_replay ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_s1c1_rsched/gmm_circle_s1c1_rsched_200k_2026-06-10_17-40-39/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_circle_s1c1_rsched/gmm_circle_s1c1_rsched_200k_2026-06-10_17-40-39/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle  --out-dir vj/rsched_replay_ep1999
echo "COV_rsched_replay_ep1999_EXIT=$?"
echo DONE_rsched
