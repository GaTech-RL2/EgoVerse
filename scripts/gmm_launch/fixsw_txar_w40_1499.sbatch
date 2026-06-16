#!/bin/bash
#SBATCH --job-name=fixsw_txar_w40_1499
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=0:45:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/fixsw_txar_w40_1499_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## txar_w40_e15p ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/txar_w40_e15p_ep1499
echo "COV_txar_w40_e15p_ep1499_EXIT=$?"
echo "DONE_txar_w40_1499"
