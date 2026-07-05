#!/bin/bash
#SBATCH --job-name=gmm_txar_obs8_fp32ext2
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=0:55:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/gmm_txar_obs8_fp32ext2_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## gmm_txar_obs8 ep1499 FP32 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --pusher circle --out-dir fp32ext2/gmm_txar_obs8_ep1499
echo "COV_gmm_txar_obs8_ep1499_EXIT=$?"
echo DONE_gmm_txar_obs8
