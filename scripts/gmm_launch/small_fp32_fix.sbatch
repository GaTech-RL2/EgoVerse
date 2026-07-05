#!/bin/bash
#SBATCH --job-name=small_fp32fix
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=0:55:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/small_fp32fix_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
RUN=logs/gmm_small_obs8/gmm_small_obs8_200k_2026-06-09_00-28-48
CFG=$RUN/.hydra/config.yaml
for ep in 499 999 1499 1999; do
  echo "########## gmm_small_obs8 ep${ep} FP32 ##########"
  .venv/bin/python -m egomimic.eval.core.ckpt_loading \
    --ckpt "$RUN/checkpoints/epoch_epoch=${ep}.ckpt" --config-path "$CFG" \
    --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 \
    --embodiment-name pushshapes_sim --pusher circle \
    --out-dir fp32sweep/gmm_small_obs8_ep${ep}
  echo "COV_gmm_small_obs8_ep${ep}_EXIT=$?"
done
echo "DONE_gmm_small_obs8"
