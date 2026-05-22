#!/bin/bash
#SBATCH --job-name=dfot-fsd-eval
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=200G
#SBATCH --time=0:30:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/logs/sbatch/fsd_eval_%j.out
#SBATCH --error=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/logs/sbatch/fsd_eval_%j.err

set -euxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3
mkdir -p logs/sbatch logs/fsd_eval

source .venv/bin/activate
export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"
export MUJOCO_GL=egl
export PYTHONPATH=.

CKPT_OLD=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/dfot_variants/dfot_200ep_full750_lr4e5_2026-05-21_00-28-28/checkpoints/epoch_epoch=199.ckpt
CFG_OLD=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/dfot_variants/dfot_200ep_full750_lr4e5_2026-05-21_00-28-28/.hydra/config.yaml

CKPT_NEW=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/dfot_variants/dfot_400ep_attndrop_2026-05-21_03-15-27/checkpoints/last.ckpt
CFG_NEW=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/dfot_variants/dfot_400ep_attndrop_2026-05-21_03-15-27/.hydra/config.yaml

echo "=== OLD ep199 FSD horizon=32 ==="
python scripts/eval_cfg_latest.py --ckpt "$CKPT_OLD" --config-path "$CFG_OLD" \
  --n-episodes 4 --max-steps 1200 --inference-mode chunk --cfg-scale 1.0 \
  --skip-val --out-dir logs/fsd_eval/OLD_fsd_h32 2>&1 | tee logs/fsd_eval/OLD_fsd_h32.log

echo "=== NEW ep399 (attndrop) FSD horizon=32 ==="
python scripts/eval_cfg_latest.py --ckpt "$CKPT_NEW" --config-path "$CFG_NEW" \
  --n-episodes 4 --max-steps 1200 --inference-mode chunk --cfg-scale 1.0 \
  --skip-val --out-dir logs/fsd_eval/NEW_fsd_h32 2>&1 | tee logs/fsd_eval/NEW_fsd_h32.log
