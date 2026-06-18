#!/bin/bash
#SBATCH --job-name=rtw_xemb_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# Inference-only: pi0.5 weights (~14.5GB fp32) + DINOv2 + SAPIEN fit a 48GB l40s,
# which schedules far faster than the contended gpu-h200.
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=06:00:00
#SBATCH --output=egomimic/ricl/outputs/robotwin_eval/xemb_compare_%j.out
#SBATCH --error=egomimic/ricl/outputs/robotwin_eval/xemb_compare_%j.err

# CLOSED-LOOP CROSS-EMBODIMENT comparison on HELD-OUT RoboTwin tasks.
# The SAPIEN sim controls embodiment B = aloha-agilex (the predicted embodiment,
# demo_clean default). For each held-out task we run three policies and compare
# closed-loop success:
#   xemb  : cross-embodiment RICL — predict aloha, retrieve ARX-X5 demos (emb A).
#   same  : same-embodiment RICL  — predict aloha, retrieve aloha demos (upper bound).
#   plain : no in-context demos    — plain aloha pi0.5 finetune (floor).
# Only the retrieval bank differs across xemb/same; the controlled robot is always
# aloha. xemb normalizes retrieved demos with the ARX-X5 bank quantiles
# (bank_quantiles_path) — exactly as in cross-embodiment training.
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
# demo_clean (default) or demo_randomized (domain randomization: cluttered table,
# random background/light/table-height — breaks the clean-task success ceiling).
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"

TRAIN="$REPO/egomimic/ricl/outputs/robotwin_train"
# Cross-embodiment RICL run (predict aloha / retrieve arx-x5).
XEMB_DIR="$TRAIN/robotwin_xemb_ricl/version_0"
XEMB_CKPT=$(ls -t "$XEMB_DIR"/checkpoints/*.ckpt | head -1)
XEMB_Q="$XEMB_DIR/quantiles.json"             # aloha (query/action) quantiles
XEMB_BANK_Q="$XEMB_DIR/bank_quantiles.json"   # arx-x5 (retrieved demo) quantiles
# Existing same-embodiment baselines (predict aloha).
SAME_CKPT=$(ls -t "$TRAIN"/robotwin_mt_ricl/version_0/checkpoints/*.ckpt | head -1)
SAME_Q="$TRAIN/robotwin_mt_ricl/version_0/quantiles.json"
PLAIN_CKPT=$(ls -t "$TRAIN"/robotwin_mt_plain/version_0/checkpoints/*.ckpt | head -1)
PLAIN_Q="$TRAIN/robotwin_mt_plain/version_0/quantiles.json"

# Held-out eval tasks that have BOTH an aloha and an arx-x5 bank (and run in sim).
TASKS="${TASKS:-click_alarmclock open_microwave turn_switch}"
ALOHA_BANK="$REPO/egomimic/ricl/outputs/robotwin_mt/eval"          # emb B banks (same-emb)
ARX_BANK="$REPO/egomimic/ricl/outputs/robotwin_xemb_bank/eval"     # emb A banks (cross-emb)
IDXBASE="$REPO/egomimic/ricl/outputs/robotwin_xemb_eval_index"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} N=$EVAL_TEST_NUM task_config=$TASK_CONFIG"
echo "XEMB_CKPT=$XEMB_CKPT"
echo "SAME_CKPT=$SAME_CKPT"
echo "PLAIN_CKPT=$PLAIN_CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_eval () {  # name task ckpt quant bank idx noic bankquant
  echo "##### EVAL model=$1 task=$2 no_incontext=$7 bank=$5 #####"
  local extra=""
  [ -n "${8:-}" ] && extra="--bank_quantiles_path $8"
  ( cd "$REPO/external/RoboTwin" && python -u script/eval_policy.py \
      --config policy/pi_ricl_egoverse/deploy_policy.yml --overrides \
      --task_name "$2" --task_config "$TASK_CONFIG" --policy_name pi_ricl_egoverse \
      --egoverse_checkpoint "$3" --quantiles_path "$4" \
      --bank_root "$5" --bank_index_dir "$6" --no_incontext "$7" $extra \
      --model_name "$1" --ckpt_setting "$1" --seed 0 ) || echo "[$1/$2] EVAL FAILED"
}

for T in $TASKS; do
  ARX_IDX="$IDXBASE/arx/$T"; ALOHA_IDX="$IDXBASE/aloha/$T"
  echo "=== build ARX-X5 bank index: $T ==="
  python egomimic/ricl/scripts/build_robotwin_bank_index.py --root "$ARX_BANK/$T" --out "$ARX_IDX" --embed dinov2 \
    || { echo "arx index build FAILED for $T"; continue; }
  echo "=== build aloha bank index: $T ==="
  python egomimic/ricl/scripts/build_robotwin_bank_index.py --root "$ALOHA_BANK/$T" --out "$ALOHA_IDX" --embed dinov2 \
    || { echo "aloha index build FAILED for $T"; continue; }
  # cross-embodiment RICL: aloha query/action quantiles + arx-x5 bank quantiles.
  run_eval xemb  "$T" "$XEMB_CKPT"  "$XEMB_Q"  "$ARX_BANK/$T"   "$ARX_IDX"   false "$XEMB_BANK_Q"
  # same-embodiment RICL upper bound: aloha bank.
  run_eval same  "$T" "$SAME_CKPT"  "$SAME_Q"  "$ALOHA_BANK/$T" "$ALOHA_IDX" false
  # plain floor: no retrieval (bank passed only for state_dim detection).
  run_eval plain "$T" "$PLAIN_CKPT" "$PLAIN_Q" "$ALOHA_BANK/$T" "$ALOHA_IDX" true
done
echo "=== xemb eval compare DONE ==="
