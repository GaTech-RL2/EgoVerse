#!/bin/bash
#SBATCH --job-name=gmm_circle_ratiosched_fp32ext2
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=0:55:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/gmm_circle_ratiosched_fp32ext2_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## gmm_circle_ratiosched ep1999 FP32 ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_ratiosched/gmm_circle_ratiosched_200k_2026-06-09_00-30-24/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_circle_ratiosched/gmm_circle_ratiosched_200k_2026-06-09_00-30-24/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --pusher circle --out-dir fp32ext2/gmm_circle_ratiosched_ep1999
echo "COV_gmm_circle_ratiosched_ep1999_EXIT=$?"
echo DONE_gmm_circle_ratiosched
