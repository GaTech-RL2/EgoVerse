#!/bin/bash
#SBATCH --job-name=sim_ol_compare
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=0:40:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs/sim_ol_compare_%j.out
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
export PYTHONPATH=.
export MUJOCO_GL=egl
RUN=logs/gmm_circle/gmm_circle_200k_2026-06-08_12-19-23
CKPT=$RUN/checkpoints/last.ckpt
CFG=$RUN/.hydra/config.yaml

# Re-plan cadence is now the ARCH property obs_stride (sim interface uniform).
# Override it for the diagnostic: obs_stride=1 (dense, matches per-frame
# training) vs obs_stride=8 (sparse = the mismatch the bug created).
# limit_val_batches now == n_episodes, so all 16 episodes actually roll out.

echo "############## obs_stride=1  (DENSE / open-loop-1, matches per-frame training) ##############"
.venv/bin/python -m egomimic.eval.core.ckpt_loading \
  --ckpt "$CKPT" --config-path "$CFG" \
  --n-episodes 16 --max-steps 400 --obs-stride 1 --seed 0 \
  --out-dir sim_ol_compare/ol1

echo "############## obs_stride=8  (SPARSE / open-loop-8 = the train/sim mismatch) ##############"
.venv/bin/python -m egomimic.eval.core.ckpt_loading \
  --ckpt "$CKPT" --config-path "$CFG" \
  --n-episodes 16 --max-steps 400 --obs-stride 8 --seed 0 \
  --out-dir sim_ol_compare/ol8

echo "############## DONE ##############"
ls -la sim_ol_compare/ol1 sim_ol_compare/ol8 2>/dev/null
