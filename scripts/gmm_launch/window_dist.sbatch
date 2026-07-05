#!/bin/bash
#SBATCH --job-name=window_dist
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/window_dist_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## w4_sb1000 ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w4/gmm_txar_w4_200k_2026-06-09_19-43-13/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w4/gmm_txar_w4_200k_2026-06-09_19-43-13/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir window_dist/w4_sb1000
echo "COV_w4_sb1000_EXIT=$?"
echo "########## w4_sb2000 ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w4/gmm_txar_w4_200k_2026-06-09_19-43-13/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w4/gmm_txar_w4_200k_2026-06-09_19-43-13/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 2000 --rng-pairing --out-dir window_dist/w4_sb2000
echo "COV_w4_sb2000_EXIT=$?"
echo "########## w4_sb3000 ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w4/gmm_txar_w4_200k_2026-06-09_19-43-13/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w4/gmm_txar_w4_200k_2026-06-09_19-43-13/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 3000 --rng-pairing --out-dir window_dist/w4_sb3000
echo "COV_w4_sb3000_EXIT=$?"
echo "########## w10_sb1000 ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w10/gmm_txar_w10_200k_2026-06-09_19-53-28/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w10/gmm_txar_w10_200k_2026-06-09_19-53-28/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir window_dist/w10_sb1000
echo "COV_w10_sb1000_EXIT=$?"
echo "########## w10_sb2000 ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w10/gmm_txar_w10_200k_2026-06-09_19-53-28/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w10/gmm_txar_w10_200k_2026-06-09_19-53-28/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 2000 --rng-pairing --out-dir window_dist/w10_sb2000
echo "COV_w10_sb2000_EXIT=$?"
echo "########## w10_sb3000 ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w10/gmm_txar_w10_200k_2026-06-09_19-53-28/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w10/gmm_txar_w10_200k_2026-06-09_19-53-28/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 3000 --rng-pairing --out-dir window_dist/w10_sb3000
echo "COV_w10_sb3000_EXIT=$?"
echo "########## w40_sb1000 ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir window_dist/w40_sb1000
echo "COV_w40_sb1000_EXIT=$?"
echo "########## w40_sb2000 ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 2000 --rng-pairing --out-dir window_dist/w40_sb2000
echo "COV_w40_sb2000_EXIT=$?"
echo "########## w40_sb3000 ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 3000 --rng-pairing --out-dir window_dist/w40_sb3000
echo "COV_w40_sb3000_EXIT=$?"
echo "########## w128_sb1000 ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir window_dist/w128_sb1000
echo "COV_w128_sb1000_EXIT=$?"
echo "########## w128_sb2000 ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 2000 --rng-pairing --out-dir window_dist/w128_sb2000
echo "COV_w128_sb2000_EXIT=$?"
echo "########## w128_sb3000 ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 3000 --rng-pairing --out-dir window_dist/w128_sb3000
echo "COV_w128_sb3000_EXIT=$?"
echo DONE_window_dist
