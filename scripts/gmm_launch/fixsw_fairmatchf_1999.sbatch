#!/bin/bash
#SBATCH --job-name=fixsw_fairmatchf_1999
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/fixsw_fairmatchf_1999_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
echo "########## fairmatchf_replay ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --out-dir fixsweep/fairmatchf_replay_ep1999
echo "COV_fairmatchf_replay_ep1999_EXIT=$?"
echo "########## fairmatchf_sb1000 ep1999 FIXED ##########"
.venv/bin/python -m egomimic.eval.core.ckpt_loading --ckpt "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/checkpoints/epoch_epoch=1999.ckpt" --config-path "logs/gmm_circle_fairmatch/gmm_circle_fairmatch_200k_2026-06-10_04-45-57/.hydra/config.yaml" --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 --embodiment-name pushshapes_sim --only-emb 15 --pusher circle --init-mode seeds --init-seed-base 1000 --rng-pairing --out-dir fixsweep/fairmatchf_sb1000_ep1999
echo "COV_fairmatchf_sb1000_ep1999_EXIT=$?"
echo "DONE_fairmatchf"
