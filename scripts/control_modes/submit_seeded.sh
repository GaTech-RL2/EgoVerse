#!/usr/bin/env bash
# Seeded relaunch of the control-mode study.
#
# WHY. The first grid ran ONE seed per arm and the comparison turned out to be
# dominated by run-to-run variance. arm3 - arm2 on fine-bucket direction
# agreement flipped sign at every epoch measured:
#
#     ep7 +0.083 | ep11 -0.198 | ep25 -0.026 | ep30 -0.181
#
# and the residual-target ablation looked reproducible across two epochs before
# a third flipped it. Nothing finer than "all arms overshoot the fine phase by
# 12-29x and sit near chance on direction" survived. Multiple seeds is the fix.
#
# WHAT CHANGED versus submit.sh, all measured problems:
#   seed=N                          three seeds per arm instead of one
#   callbacks=checkpoints_monitored ModelCheckpoint now tracks Valid/action_loss
#                                   (previously monitor=None -> best-checkpoint
#                                   selection never happened, and held-out loss
#                                   stops improving ~epoch 7-11 while train loss
#                                   keeps falling, so last.ckpt was the WORST
#                                   checkpoint for generalisation)
#   model.enable_grad_norm=true     enables the MAD-based adaptive gradient
#                                   spike clipping in on_after_backward, which
#                                   was disabled so no clipping ran at all
#   MAX_EPOCHS default 40           held-out loss bottoms by ~epoch 11; 100
#                                   epochs bought overfitting and 60h of GPU
#
# SCOPE. Small capacity only, arms 1-3, three seeds = 9 jobs. Small because the
# large arms train 2-15x slower and arm1-large never finished (46 min/epoch,
# 76h projected). Arms 1-3 because arm4 (state_idm) was the one arm clearly
# WORSE than the others at every epoch measured, so it is not where the
# resolution is needed. arm3 vs arm2 IS the study's headline.
#
# Usage:
#   scripts/control_modes/submit_seeded.sh [--dry-run]
# Env overrides: SEEDS, ARMS, POOL, NUM_GPU, MAX_EPOCHS, BRANCH, PRIORITY
set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="$REPO/osmo/pushshapes_control_modes_seeded.yaml"
test -s "$LAUNCHER" || { echo "missing launcher: $LAUNCHER" >&2; exit 1; }

BRANCH="${BRANCH:-algo/causal-action-models}"
POOL="${POOL:-groot-l40-01}"
PRIORITY="${PRIORITY:-HIGH}"
# 1 GPU per job: groot-l40-01 caps CPUs at 1/8 of the node per GPU (15), so a
# larger CPU request is rejected outright with "CPU value too high".
NUM_GPU="${NUM_GPU:-1}"
CPU="${CPU:-12}"
MEMORY="${MEMORY:-100Gi}"
STORAGE="${STORAGE:-150Gi}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-2500}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
WANDB_ENTITY="${WANDB_ENTITY:-rl2-group}"
SEEN_MODES="${SEEN_MODES:-tight loose laggy sticky}"
DATA_PREFIX="${DATA_PREFIX:-s3://rldb/processed_v3/pushshapes_sim/control_gap_dedup_gripper_simv2_20260830}"
N_PER_MODE="${N_PER_MODE:-547}"
DATA_CFG="${DATA_CFG:-pusht/control_modes_gripper_arc_D10_M16_append_r0}"
STAMP="$(date +%Y%m%d)"

read -r -a SEEDS <<< "${SEEDS:-42 1337 2718}"
read -r -a ARMS <<< "${ARMS:-arm1_dp_flow arm2_causal_bidir arm3_state_action_ar}"

arm_desc() {
  case "$1" in
    arm1_dp_flow)         echo "flow-bidir-seeded" ;;
    arm2_causal_bidir)    echo "MSE-bidir-CONTROL-seeded" ;;
    arm3_state_action_ar) echo "MSE-causal-AR-seeded" ;;
    arm4_state_idm)       echo "MSE-causal-IDM-seeded" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

n=0
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    MODEL_CFG="bf/bf_ctrlmode_${arm}_small"
    JOB="ctrlseed-${arm//_/-}-s${seed}"
    RUN_NAME="ctrlseed_${arm}_s${seed}_${STAMP}"
    RUN_DESC="$(arm_desc "$arm")-s${seed}"

    # wandb builds its run id as name_description_timestamp and rejects >128
    # chars in PHASE 2, after norm-stats has already run. Check the composed
    # length rather than eyeballing the parts.
    WANDB_ID_LEN=$(( ${#RUN_NAME} + 1 + ${#RUN_DESC} + 1 + 19 ))
    if [ "$WANDB_ID_LEN" -gt 128 ]; then
      echo "wandb id would be $WANDB_ID_LEN chars (>128): $RUN_NAME/$RUN_DESC" >&2
      exit 2
    fi

    ARGS=(
      workflow submit "$LAUNCHER"
      --pool "$POOL" --priority "$PRIORITY" --set
      "job_name=$JOB" "branch=$BRANCH"
      "data_cfg=$DATA_CFG" "model_cfg=$MODEL_CFG"
      "run_name=$RUN_NAME" "run_desc=$RUN_DESC"
      "seed=$seed"
      "num_gpu=$NUM_GPU" "batch_size=$BATCH_SIZE" "num_workers=$NUM_WORKERS"
      "max_epochs=$MAX_EPOCHS" "limit_train_batches=$LIMIT_TRAIN_BATCHES"
      "wandb_entity=$WANDB_ENTITY" "seen_modes=$SEEN_MODES"
      "data_prefix=$DATA_PREFIX" "n_per_mode=$N_PER_MODE"
      "cpu=$CPU" "memory=$MEMORY" "storage=$STORAGE"
    )

    echo "=== $JOB ($MODEL_CFG, seed=$seed) ==="
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "  osmo ${ARGS[*]}"
    else
      osmo "${ARGS[@]}" 2>&1 \
        | grep -aE "Workflow ID|error|Error|failed|Assertion" | head -3
    fi
    n=$((n+1))
  done
done
echo "submitted $n jobs (${#ARMS[@]} arms x ${#SEEDS[@]} seeds)"
