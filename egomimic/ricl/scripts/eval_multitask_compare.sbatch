#!/bin/bash
#SBATCH --job-name=rtw_mt_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# Eval is inference-only: pi0.5 weights (~14.5GB fp32) fit a 48GB l40s with room for
# DINOv2 + SAPIEN, and gpu-l40s schedules far faster than the contended gpu-h200.
# (Override to gpu-h200/gpu-a100 via sbatch -p/--gres if l40s is drained.)
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --output=egomimic/ricl/outputs/robotwin_eval/mt_compare_%j.out
#SBATCH --error=egomimic/ricl/outputs/robotwin_eval/mt_compare_%j.err

# Closed-loop RICL-vs-plain comparison on HELD-OUT RoboTwin tasks.
# For each held-out task: build its DINOv2 bank index, then eval the RICL model
# (within-task retrieval ON) and the plain model (retrieval OFF). The RICL model
# never trained on these tasks — it only gets k=4 in-context demos at eval.
# Queue after training: sbatch --dependency=afterany:<ricl_jid>:<plain_jid> ...
set -uo pipefail
REPO=/storage/project/r-dxu345-0/rco3/EgoVerse2
cd "$REPO"
mkdir -p egomimic/ricl/outputs/robotwin_eval
source emimic/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export PYTHONWARNINGS=ignore::UserWarning
export DISPLAY=""
export EVAL_TEST_NUM="${EVAL_TEST_NUM:-10}"
# demo_clean (default) or demo_randomized (cluttered table, random bg/light/table-height).
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"

MT="$REPO/egomimic/ricl/outputs/robotwin_mt"
TRAIN="$REPO/egomimic/ricl/outputs/robotwin_train"
IDXBASE="$REPO/egomimic/ricl/outputs/robotwin_mt_eval_index"
RICL_CKPT=$(ls -t "$TRAIN"/robotwin_mt_ricl/version_0/checkpoints/*.ckpt | head -1)
RICL_Q="$TRAIN/robotwin_mt_ricl/version_0/quantiles.json"
PLAIN_CKPT=$(ls -t "$TRAIN"/robotwin_mt_plain/version_0/checkpoints/*.ckpt | head -1)
PLAIN_Q="$TRAIN/robotwin_mt_plain/version_0/quantiles.json"
TASKS="${TASKS:-click_alarmclock open_microwave}"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} N=$EVAL_TEST_NUM task_config=$TASK_CONFIG"
echo "RICL_CKPT=$RICL_CKPT"
echo "PLAIN_CKPT=$PLAIN_CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_eval () {  # name task ckpt quant bank idx noic
  echo "##### EVAL model=$1 task=$2 no_incontext=$7 #####"
  ( cd "$REPO/external/RoboTwin" && python -u script/eval_policy.py \
      --config policy/pi_ricl_egoverse/deploy_policy.yml --overrides \
      --task_name "$2" --task_config "$TASK_CONFIG" --policy_name pi_ricl_egoverse \
      --egoverse_checkpoint "$3" --quantiles_path "$4" \
      --bank_root "$5" --bank_index_dir "$6" --no_incontext "$7" \
      --model_name "$1" --ckpt_setting "$1" --seed 0 ) || echo "[$1/$2] EVAL FAILED"
}

for T in $TASKS; do
  BANK="$MT/eval/$T"
  IDX="$IDXBASE/$T"
  echo "=== build bank index: $T ==="
  python egomimic/ricl/scripts/build_robotwin_bank_index.py --root "$BANK" --out "$IDX" --embed dinov2 \
    || { echo "index build FAILED for $T"; continue; }
  run_eval ricl  "$T" "$RICL_CKPT"  "$RICL_Q"  "$BANK" "$IDX" false
  run_eval plain "$T" "$PLAIN_CKPT" "$PLAIN_Q" "$BANK" "$IDX" true
done
echo "=== mt eval compare DONE ==="
