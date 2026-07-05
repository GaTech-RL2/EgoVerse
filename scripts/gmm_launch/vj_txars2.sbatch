#!/bin/bash
#SBATCH --job-name=vj_txars2
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/vj_txars2_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## txars2_seeds ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_s2/gmm_txar_s2_200k_2026-06-10_13-27-47/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_s2/gmm_txar_s2_200k_2026-06-10_13-27-47/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir vj/txars2_seeds_ep1499
echo "COV_txars2_seeds_ep1499_EXIT=$?"
echo "########## txars2_replay ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_s2/gmm_txar_s2_200k_2026-06-10_13-27-47/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_s2/gmm_txar_s2_200k_2026-06-10_13-27-47/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle  --out-dir vj/txars2_replay_ep1499
echo "COV_txars2_replay_ep1499_EXIT=$?"
echo DONE_txars2
