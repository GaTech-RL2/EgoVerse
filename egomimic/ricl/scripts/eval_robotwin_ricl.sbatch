#!/bin/bash
#SBATCH --job-name=robotwin_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ig-88
#SBATCH --time=01:00:00
#SBATCH --output=egomimic/ricl/outputs/robotwin_eval/slurm_%j.out
#SBATCH --error=egomimic/ricl/outputs/robotwin_eval/slurm_%j.err

# Closed-loop eval of the EgoVerse pi0.5 RICL checkpoint on a RoboTwin task.
# Replicates policy/pi_ricl_egoverse/eval.sh but uses the (uv-managed) emimic venv
# and the correct cwd. Bank/quantiles/checkpoint paths come from deploy_policy.yml.
#   TASK / TASK_CONFIG / CKPT / SEED overridable via env vars.
set -uo pipefail
mkdir -p egomimic/ricl/outputs/robotwin_eval
cd /coc/flash7/rco3/EgoVerse
source emimic/bin/activate

TASK="${TASK:-beat_block_hammer}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
CKPT="${CKPT:-egomimic/ricl/outputs/robotwin_train/robotwin_ricl/version_2/checkpoints/last.ckpt}"
CKPT_ABS="$(readlink -f "$CKPT")"
SEED="${SEED:-0}"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} ckpt=$CKPT_ABS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# SAPIEN headless offscreen rendering on the compute node
export DISPLAY=""
export PYTHONWARNINGS=ignore::UserWarning
# Eval-only: run eager (pi0.5 sample_actions triggers a multi-minute torch.compile
# max-autotune otherwise). Bound the episode count for a fast eval (default 100).
export TORCH_COMPILE_DISABLE=1
export EVAL_TEST_NUM="${EVAL_TEST_NUM:-10}"

cd external/RoboTwin
python script/eval_policy.py --config policy/pi_ricl_egoverse/deploy_policy.yml \
  --overrides \
  --task_name "${TASK}" \
  --task_config "${TASK_CONFIG}" \
  --egoverse_checkpoint "${CKPT_ABS}" \
  --model_name eva_bimanual \
  --ckpt_setting eva_bimanual \
  --seed "${SEED}" \
  --policy_name pi_ricl_egoverse

echo "=== eval exit: $? ==="
