#!/bin/bash
#SBATCH --job-name=hpt_eval_sweep
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=3:30:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/hpt_eval_sweep_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## hpt_rp32 ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_hpt_fixed/gmm_hpt_fixed_200k_2026-06-09_20-14-23/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_hpt_fixed/gmm_hpt_fixed_200k_2026-06-09_20-14-23/.hydra/config.yaml" --eval-class hpt --replan-every 32 --n-episodes 16 --max-steps 400 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/hpt_rp32_ep1999
echo "COV_hpt_rp32_ep1999_EXIT=$?"
echo "########## hpt_rp8 ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_hpt_fixed/gmm_hpt_fixed_200k_2026-06-09_20-14-23/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_hpt_fixed/gmm_hpt_fixed_200k_2026-06-09_20-14-23/.hydra/config.yaml" --eval-class hpt --replan-every 8 --n-episodes 16 --max-steps 400 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/hpt_rp8_ep1999
echo "COV_hpt_rp8_ep1999_EXIT=$?"
echo "########## hpt_rp1 ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_hpt_fixed/gmm_hpt_fixed_200k_2026-06-09_20-14-23/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_hpt_fixed/gmm_hpt_fixed_200k_2026-06-09_20-14-23/.hydra/config.yaml" --eval-class hpt --replan-every 1 --n-episodes 16 --max-steps 400 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/hpt_rp1_ep1999
echo "COV_hpt_rp1_ep1999_EXIT=$?"
echo "########## hpt_rp32 ep1499 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_hpt_fixed/gmm_hpt_fixed_200k_2026-06-09_20-14-23/checkpoints/epoch_epoch=1499.ckpt" --config-path "logs/gmm_hpt_fixed/gmm_hpt_fixed_200k_2026-06-09_20-14-23/.hydra/config.yaml" --eval-class hpt --replan-every 32 --n-episodes 16 --max-steps 400 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/hpt_rp32_ep1499
echo "COV_hpt_rp32_ep1499_EXIT=$?"
echo DONE_hpt_sweep
