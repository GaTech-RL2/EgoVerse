#!/bin/bash
#SBATCH --job-name=dfot-sim-sweep
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/logs/sbatch/sim_sweep_%j.out
#SBATCH --error=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/logs/sbatch/sim_sweep_%j.err

# Apply offline AR-sweep findings to closed-loop sim on:
#   (A) ep199 of original 200ep run (baseline 0.0075)
#   (B) ep399 of 400ep+attndrop run (baseline 0.0424)
# Each ckpt × multiple (ar_inference_chunk_size, ar_inference_step_size) combos.
# Run in EgoVerse-clone-3 (has the patched _inference_step_ar with step_size).

set -euxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3
mkdir -p logs/sbatch logs/closedloop_sweep

source .venv/bin/activate
export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"
export MUJOCO_GL=egl
export PYTHONPATH=.

CKPT_OLD=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/dfot_variants/dfot_200ep_full750_lr4e5_2026-05-21_00-28-28/checkpoints/epoch_epoch=199.ckpt
CFG_OLD=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/dfot_variants/dfot_200ep_full750_lr4e5_2026-05-21_00-28-28/.hydra/config.yaml

CKPT_NEW=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/dfot_variants/dfot_400ep_attndrop_2026-05-21_03-15-27/checkpoints/last.ckpt
CFG_NEW=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse4/logs/dfot_variants/dfot_400ep_attndrop_2026-05-21_03-15-27/.hydra/config.yaml

# Pair of ckpt+label
for run in "OLD:$CKPT_OLD:$CFG_OLD" "NEW:$CKPT_NEW:$CFG_NEW"; do
  LABEL=${run%%:*}
  rest=${run#*:}
  CKPT=${rest%%:*}
  CFG=${rest#*:}
  for cs_ss in 4_4 8_4 16_4 4_8; do
    CS=${cs_ss%_*}; SS=${cs_ss#*_}
    OUT_DIR=logs/closedloop_sweep/${LABEL}_chunk${CS}_step${SS}
    echo "=== ${LABEL} chunk=${CS} step=${SS} ==="
    python scripts/eval_cfg_latest.py \
      --ckpt "$CKPT" --config-path "$CFG" \
      --n-episodes 4 --max-steps 1200 \
      --inference-mode ar --cfg-scale 1.0 \
      --ar-inference-chunk-size $CS --ar-inference-step-size $SS \
      --skip-val \
      --out-dir "$OUT_DIR" \
      2>&1 | tee "${OUT_DIR}.log"
  done
done
