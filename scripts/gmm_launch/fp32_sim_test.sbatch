#!/bin/bash
#SBATCH --job-name=fp32_sim_test
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=0:35:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/fp32_sim_test_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
RUN=logs/gmm_circle_obs8/gmm_circle_obs8_200k_2026-06-09_00-28-15
CFG=$RUN/.hydra/config.yaml

echo "############## FP32 SIM, ep999 ckpt (bf16 training-val was 0.091) ##############"
.venv/bin/python -m egomimic.eval.core.ckpt_loading \
  --ckpt "$RUN/checkpoints/epoch_epoch=999.ckpt" --config-path "$CFG" \
  --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 \
  --out-dir fp32sim/ep999
echo "EXIT_999=$?"

echo "############## FP32 SIM, ep1499 ckpt (bf16 training-val was 0.117) ##############"
.venv/bin/python -m egomimic.eval.core.ckpt_loading \
  --ckpt "$RUN/checkpoints/epoch_epoch=1499.ckpt" --config-path "$CFG" \
  --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 \
  --out-dir fp32sim/ep1499
echo "EXIT_1499=$?"
echo "############## DONE ##############"
