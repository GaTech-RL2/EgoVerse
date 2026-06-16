#!/bin/bash
#SBATCH --job-name=fixsw_nogen
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/fixsw_nogen_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## nogen_e17 ep499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_cotrain_nogen_obs8/gmm_cotrain_nogen_obs8_200k_2026-06-09_00-48-55/checkpoints/epoch_epoch=499.ckpt" --config-path "logs/gmm_cotrain_nogen_obs8/gmm_cotrain_nogen_obs8_200k_2026-06-09_00-48-55/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 17 --pusher circle_small --out-dir fixsweep/nogen_e17_ep499
echo "COV_nogen_e17_ep499_EXIT=$?"
echo "########## nogen_e17 ep999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_cotrain_nogen_obs8/gmm_cotrain_nogen_obs8_200k_2026-06-09_00-48-55/checkpoints/epoch_epoch=999.ckpt" --config-path "logs/gmm_cotrain_nogen_obs8/gmm_cotrain_nogen_obs8_200k_2026-06-09_00-48-55/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 17 --pusher circle_small --out-dir fixsweep/nogen_e17_ep999
echo "COV_nogen_e17_ep999_EXIT=$?"
echo "########## nogen_e15p ep499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_cotrain_nogen_obs8/gmm_cotrain_nogen_obs8_200k_2026-06-09_00-48-55/checkpoints/epoch_epoch=499.ckpt" --config-path "logs/gmm_cotrain_nogen_obs8/gmm_cotrain_nogen_obs8_200k_2026-06-09_00-48-55/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/nogen_e15p_ep499
echo "COV_nogen_e15p_ep499_EXIT=$?"
echo "########## nogen_e15p ep999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_cotrain_nogen_obs8/gmm_cotrain_nogen_obs8_200k_2026-06-09_00-48-55/checkpoints/epoch_epoch=999.ckpt" --config-path "logs/gmm_cotrain_nogen_obs8/gmm_cotrain_nogen_obs8_200k_2026-06-09_00-48-55/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/nogen_e15p_ep999
echo "COV_nogen_e15p_ep999_EXIT=$?"
echo DONE_fixsw_nogen
