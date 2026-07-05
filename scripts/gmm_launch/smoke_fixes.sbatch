#!/bin/bash
#SBATCH --job-name=smoke_evalfixes
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=0:40:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/smoke_evalfixes_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
CIRCLE=logs/gmm_circle_obs8/gmm_circle_obs8_200k_2026-06-09_00-28-15
COTRAIN=logs/gmm_cotrain_obs8/gmm_cotrain_obs8_200k_2026-06-09_00-48-55

echo "########## A1: seeds+pairing run1 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "$CIRCLE/checkpoints/epoch_epoch=1999.ckpt" --config-path "$CIRCLE/.hydra/config.yaml" --n-episodes 2 --max-steps 60 --obs-stride 8 --seed 0 --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir smoke_fixes/a1
echo "A1_EXIT=$?"
echo "########## A2: seeds+pairing run2 (must match A1) ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "$CIRCLE/checkpoints/epoch_epoch=1999.ckpt" --config-path "$CIRCLE/.hydra/config.yaml" --n-episodes 2 --max-steps 60 --obs-stride 8 --seed 0 --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir smoke_fixes/a2
echo "A2_EXIT=$?"
echo "########## B: only-emb 17 + circle_small (port) ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "$COTRAIN/checkpoints/epoch_epoch=999.ckpt" --config-path "$COTRAIN/.hydra/config.yaml" --n-episodes 2 --max-steps 60 --obs-stride 8 --seed 0 --only-emb 17 --pusher circle_small --out-dir smoke_fixes/b
echo "B_EXIT=$?"
echo "########## C: replay regression (old default path) ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "$CIRCLE/checkpoints/epoch_epoch=1999.ckpt" --config-path "$CIRCLE/.hydra/config.yaml" --n-episodes 2 --max-steps 60 --obs-stride 8 --seed 0 --out-dir smoke_fixes/c
echo "C_EXIT=$?"
echo "DONE_SMOKE_FIXES"
