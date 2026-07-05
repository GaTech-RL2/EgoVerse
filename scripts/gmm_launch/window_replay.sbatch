#!/bin/bash
#SBATCH --job-name=window_replay
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/window_replay_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## w4_replay ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w4/gmm_txar_w4_200k_2026-06-09_19-43-13/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w4/gmm_txar_w4_200k_2026-06-09_19-43-13/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir window_replay/w4
echo "COV_w4_replay_EXIT=$?"
echo "########## w10_replay ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w10/gmm_txar_w10_200k_2026-06-09_19-53-28/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w10/gmm_txar_w10_200k_2026-06-09_19-53-28/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir window_replay/w10
echo "COV_w10_replay_EXIT=$?"
echo "########## w40_replay ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir window_replay/w40
echo "COV_w40_replay_EXIT=$?"
echo "########## w128_replay ep1499 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_txar_obs8/gmm_txar_obs8_200k_2026-06-09_00-29-50/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir window_replay/w128
echo "COV_w128_replay_EXIT=$?"
echo DONE_window_replay
