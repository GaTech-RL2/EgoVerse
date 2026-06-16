#!/bin/bash
#SBATCH --job-name=fixsw_wide_anchors
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/fixsw_wide_anchors_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## txar_sb2000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 2000 --rng-pairing --out-dir fixsweep/txar_sb2000_ep1499
echo "COV_txar_sb2000_ep1499_EXIT=$?"
echo "########## txar_sb3000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 3000 --rng-pairing --out-dir fixsweep/txar_sb3000_ep1499
echo "COV_txar_sb3000_ep1499_EXIT=$?"
echo "########## txar_sb2000 ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 2000 --rng-pairing --out-dir fixsweep/txar_sb2000_ep1999
echo "COV_txar_sb2000_ep1999_EXIT=$?"
echo "########## txar_sb3000 ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 3000 --rng-pairing --out-dir fixsweep/txar_sb3000_ep1999
echo "COV_txar_sb3000_ep1999_EXIT=$?"
echo "########## circle_sb2000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_obs8/gmm_circle_obs8_200k_2026-06-09_00-28-15/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_obs8/gmm_circle_obs8_200k_2026-06-09_00-28-15/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 2000 --rng-pairing --out-dir fixsweep/circle_sb2000_ep1499
echo "COV_circle_sb2000_ep1499_EXIT=$?"
echo "########## circle_sb3000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_obs8/gmm_circle_obs8_200k_2026-06-09_00-28-15/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_obs8/gmm_circle_obs8_200k_2026-06-09_00-28-15/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 3000 --rng-pairing --out-dir fixsweep/circle_sb3000_ep1499
echo "COV_circle_sb3000_ep1499_EXIT=$?"
echo "########## circle_sb2000 ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_obs8/gmm_circle_obs8_200k_2026-06-09_00-28-15/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_circle_obs8/gmm_circle_obs8_200k_2026-06-09_00-28-15/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 2000 --rng-pairing --out-dir fixsweep/circle_sb2000_ep1999
echo "COV_circle_sb2000_ep1999_EXIT=$?"
echo "########## circle_sb3000 ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_obs8/gmm_circle_obs8_200k_2026-06-09_00-28-15/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_circle_obs8/gmm_circle_obs8_200k_2026-06-09_00-28-15/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 3000 --rng-pairing --out-dir fixsweep/circle_sb3000_ep1999
echo "COV_circle_sb3000_ep1999_EXIT=$?"
echo DONE_wide_anchors
