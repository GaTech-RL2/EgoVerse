#!/bin/bash
#SBATCH --job-name=vj_fms2
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=embers
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/vj_fms2_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## fms2_seeds ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_fairmatch_s2/gmm_circle_fairmatch_s2_200k_2026-06-10_13-23-49/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_fairmatch_s2/gmm_circle_fairmatch_s2_200k_2026-06-10_13-23-49/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir vj/fms2_seeds_ep1499
echo "COV_fms2_seeds_ep1499_EXIT=$?"
echo "########## fms2_replay ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_fairmatch_s2/gmm_circle_fairmatch_s2_200k_2026-06-10_13-23-49/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_fairmatch_s2/gmm_circle_fairmatch_s2_200k_2026-06-10_13-23-49/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle  --out-dir vj/fms2_replay_ep1499
echo "COV_fms2_replay_ep1499_EXIT=$?"
echo "########## fms2_seeds ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_fairmatch_s2/gmm_circle_fairmatch_s2_200k_2026-06-10_13-23-49/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_circle_fairmatch_s2/gmm_circle_fairmatch_s2_200k_2026-06-10_13-23-49/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir vj/fms2_seeds_ep1999
echo "COV_fms2_seeds_ep1999_EXIT=$?"
echo "########## fms2_replay ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_fairmatch_s2/gmm_circle_fairmatch_s2_200k_2026-06-10_13-23-49/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_circle_fairmatch_s2/gmm_circle_fairmatch_s2_200k_2026-06-10_13-23-49/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle  --out-dir vj/fms2_replay_ep1999
echo "COV_fms2_replay_ep1999_EXIT=$?"
echo DONE_fms2
