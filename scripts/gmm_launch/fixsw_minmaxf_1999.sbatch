#!/bin/bash
#SBATCH --job-name=fixsw_minmaxf_1999
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/fixsw_minmaxf_1999_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## minmaxf_replay ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir fixsweep/minmaxf_replay_ep1999
echo "COV_minmaxf_replay_ep1999_EXIT=$?"
echo "########## minmaxf_sb1000 ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_circle_minmax/gmm_circle_minmax_200k_2026-06-10_03-39-12/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/minmaxf_sb1000_ep1999
echo "COV_minmaxf_sb1000_ep1999_EXIT=$?"
echo "DONE_minmaxf_1999"
