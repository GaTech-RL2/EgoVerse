#!/bin/bash
#SBATCH --job-name=fixsw_dist
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/fixsw_dist_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## dist_txar_sb1000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/dist_txar_sb1000
echo "COV_dist_txar_sb1000_EXIT=$?"
echo "########## dist_txar_sb2000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 2000 --rng-pairing --out-dir fixsweep/dist_txar_sb2000
echo "COV_dist_txar_sb2000_EXIT=$?"
echo "########## dist_txar_sb3000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 3000 --rng-pairing --out-dir fixsweep/dist_txar_sb3000
echo "COV_dist_txar_sb3000_EXIT=$?"
echo "########## dist_fairmatch_sb1000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/dist_fairmatch_sb1000
echo "COV_dist_fairmatch_sb1000_EXIT=$?"
echo "########## dist_fairmatch_sb2000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 2000 --rng-pairing --out-dir fixsweep/dist_fairmatch_sb2000
echo "COV_dist_fairmatch_sb2000_EXIT=$?"
echo "########## dist_fairmatch_sb3000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 3000 --rng-pairing --out-dir fixsweep/dist_fairmatch_sb3000
echo "COV_dist_fairmatch_sb3000_EXIT=$?"
echo "########## dist_minmax_sb1000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/dist_minmax_sb1000
echo "COV_dist_minmax_sb1000_EXIT=$?"
echo "########## dist_minmax_sb2000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 2000 --rng-pairing --out-dir fixsweep/dist_minmax_sb2000
echo "COV_dist_minmax_sb2000_EXIT=$?"
echo "########## dist_minmax_sb3000 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 3000 --rng-pairing --out-dir fixsweep/dist_minmax_sb3000
echo "COV_dist_minmax_sb3000_EXIT=$?"
echo DONE_dist
