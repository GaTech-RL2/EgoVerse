#!/bin/bash
#SBATCH --job-name=fixsw_win_bc1
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=2:30:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/fixsw_win_bc1_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## txar_w4_replay ep499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w4/gmm_txar_w4_200k_2026-06-09_19-43-13/checkpoints/epoch_epoch=499.ckpt" --config-path "logs/gmm_txar_w4/gmm_txar_w4_200k_2026-06-09_19-43-13/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir fixsweep/txar_w4_replay_ep499
echo "COV_txar_w4_replay_ep499_EXIT=$?"
echo "########## txar_w4_replay ep999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w4/gmm_txar_w4_200k_2026-06-09_19-43-13/checkpoints/epoch_epoch=999.ckpt" --config-path "logs/gmm_txar_w4/gmm_txar_w4_200k_2026-06-09_19-43-13/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir fixsweep/txar_w4_replay_ep999
echo "COV_txar_w4_replay_ep999_EXIT=$?"
echo "########## txar_w10_replay ep499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w10/gmm_txar_w10_200k_2026-06-09_19-53-28/checkpoints/epoch_epoch=499.ckpt" --config-path "logs/gmm_txar_w10/gmm_txar_w10_200k_2026-06-09_19-53-28/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir fixsweep/txar_w10_replay_ep499
echo "COV_txar_w10_replay_ep499_EXIT=$?"
echo "########## txar_w10_replay ep999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w10/gmm_txar_w10_200k_2026-06-09_19-53-28/checkpoints/epoch_epoch=999.ckpt" --config-path "logs/gmm_txar_w10/gmm_txar_w10_200k_2026-06-09_19-53-28/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir fixsweep/txar_w10_replay_ep999
echo "COV_txar_w10_replay_ep999_EXIT=$?"
echo "########## txar_w40_replay ep499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/checkpoints/epoch_epoch=499.ckpt" --config-path "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir fixsweep/txar_w40_replay_ep499
echo "COV_txar_w40_replay_ep499_EXIT=$?"
echo "########## txar_w40_replay ep999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/checkpoints/epoch_epoch=999.ckpt" --config-path "logs/gmm_txar_w40/gmm_txar_w40_200k_2026-06-09_19-55-05/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir fixsweep/txar_w40_replay_ep999
echo "COV_txar_w40_replay_ep999_EXIT=$?"
echo DONE_win_bc1
