#!/usr/bin/env bash
# Submit the control-mode causal study: 4 arms x 2 capacities.
#
#   trained on : tight, loose, laggy, sticky           (gripper only)
#   held out   : ideal AND jittery, both by rollout SR under their control gap
#
# The two holdouts sit on opposite sides of the training sensing-noise range
# (0.3-0.8): ideal at 0.0, jittery at 2.5. So generalization is bracketed
# rather than probed in one direction. `ideal` is a holdout because its
# generated cell was 714/1000 unreadable, not by design; when a clean one lands
# it can move back into training.
#
# SMALL is submitted first on purpose. Dedupe-only gives ~2,188 training
# episodes, which puts the large arm at ~143k params/episode — so if only half
# the grid runs, the half that lands should be the one whose result is not
# dominated by overfitting.
#
# Usage:
#   scripts/control_modes/submit.sh [--dry-run] [capacity ...]
#     capacity: small | large   (default: both, small first)
#
# Env overrides: POOL, PRIORITY, NUM_GPU, BATCH_SIZE, MAX_EPOCHS, BRANCH,
#                DATA_PREFIX, N_PER_MODE, WANDB_ENTITY
set -euo pipefail

DRY_RUN=0
CAPS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    small|large) CAPS+=("$arg") ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
[ ${#CAPS[@]} -eq 0 ] && CAPS=(small large)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="$REPO/osmo/pushshapes_control_modes_l40s.yaml"

BRANCH="${BRANCH:-algo/causal-action-models}"
# groot-h100-01 allocates WHOLE NODES: it rejects anything but num_gpu=8
# ("Assertion failed for task train: GPU value must be 8"). groot-l40s-01 is
# 4 GPU/node and rejects 8, so POOL and NUM_GPU have to move together.
POOL="${POOL:-groot-h100-01}"
PRIORITY="${PRIORITY:-HIGH}"
NUM_GPU="${NUM_GPU:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# An "epoch" is capped at LIMIT_TRAIN_BATCHES steps so the sim evaluator runs
# on a useful cadence: a natural epoch is ~19.5k steps at batch 32, which would
# yield only a handful of SR points overnight. 2500 steps is roughly a
# train:eval time split near 1:1 for the large arms.
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-2500}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
WANDB_ENTITY="${WANDB_ENTITY:-rl2-group}"
SEEN_MODES="${SEEN_MODES:-tight loose laggy sticky}"
DATA_PREFIX="${DATA_PREFIX:-s3://rldb/processed_v3/pushshapes_sim/control_gap_dedup_gripper_simv2_20260830}"
# Matches the cap the dedupe-only upload was built with. Defensive: a cap that
# is silently ignored is worse than no cap, because the run still looks right.
N_PER_MODE="${N_PER_MODE:-547}"
# GROUP-PREFIXED. `data/pusht` and `model/bf` are nested config GROUPS, so
# hydra needs `data=pusht/<name>` and `model=bf/<name>`. Without the prefix it
# fails with "Could not find 'data/<name>'" during phase-1 composition — after
# the image pull, uv sync, the R2 pull and staging. Loading the YAML by path
# succeeds either way, which is why a path-based config test cannot catch this.
DATA_CFG="${DATA_CFG:-pusht/control_modes_gripper_arc_D10_M16_append_r0}"
STAMP="$(date +%Y%m%d)"

# ARMS is overridable so a single arm can be submitted first as a canary:
# a launcher bug fails identically for all eight, and finding that out from
# one run costs one slot instead of the night.
#   ARMS="arm2_causal_bidir" scripts/control_modes/submit.sh small
read -r -a ARMS <<< "${ARMS:-arm1_dp_flow arm2_causal_bidir arm3_state_action_ar arm4_state_idm}"

# A case, not an associative array: macOS ships bash 3.2, where `declare -A`
# does not exist and this script could not be dry-run locally.
# NO COMMAS in any of these. hydra parses a comma in an override value as a
# LIST and aborts config composition with "Ambiguous value for argument", which
# happens in phase 1 — i.e. after the image pull, uv sync, the R2 pull and
# staging. Bash quoting does not help: the value reaches hydra intact and hydra
# is the one that objects. Colons are avoided for the same reason.
arm_desc() {
  case "$1" in
    arm1_dp_flow)         echo "ARM1 baseline bidirectional flow-matching" ;;
    arm2_causal_bidir)    echo "ARM2 CONTROL bidirectional regression+MSE same backbone as arms 3-4" ;;
    arm3_state_action_ar) echo "ARM3 causal AR over arc-token rows regression+MSE" ;;
    arm4_state_idm)       echo "ARM4 causal pose path + inverse dynamics regression+MSE" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

for cap in "${CAPS[@]}"; do
  for arm in "${ARMS[@]}"; do
    MODEL_CFG="bf/bf_ctrlmode_${arm}_${cap}"
    JOB="ctrlmode-${arm//_/-}-${cap}"
    RUN_NAME="ctrlmode_${arm}_${cap}_${STAMP}"
    RUN_DESC="$(arm_desc "$arm") | ${cap} capacity | train ${SEEN_MODES// /+} | holdout ideal+jittery"

    ARGS=(
      workflow submit "$LAUNCHER"
      --pool "$POOL"
      --priority "$PRIORITY"
      --set
      "job_name=$JOB"
      "branch=$BRANCH"
      "data_cfg=$DATA_CFG"
      "model_cfg=$MODEL_CFG"
      "run_name=$RUN_NAME"
      "run_desc=$RUN_DESC"
      "num_gpu=$NUM_GPU"
      "batch_size=$BATCH_SIZE"
      "num_workers=$NUM_WORKERS"
      "max_epochs=$MAX_EPOCHS"
      "limit_train_batches=$LIMIT_TRAIN_BATCHES"
      "wandb_entity=$WANDB_ENTITY"
      "seen_modes=$SEEN_MODES"
      "data_prefix=$DATA_PREFIX"
      "n_per_mode=$N_PER_MODE"
      "cpu=${CPU:-64}"
      "memory=${MEMORY:-320Gi}"
      "storage=${STORAGE:-200Gi}"
    )

    # Guard rather than trust: a comma anywhere in a --set value becomes a
    # hydra list and kills the run in phase 1, ~40 minutes of staging in.
    for a in "${ARGS[@]}"; do
      case "$a" in
        *=*,*) echo "FATAL: comma in override value, hydra will read it as a list: $a" >&2; exit 3 ;;
      esac
    done

    echo "=== $JOB ($MODEL_CFG) ==="
    if [ "$DRY_RUN" -eq 1 ]; then
      printf '  osmo'; printf ' %q' "${ARGS[@]}"; printf '\n'
      continue
    fi
    # Submissions intermittently return 503; retry rather than lose the slot.
    for attempt in 1 2 3 4 5; do
      if osmo "${ARGS[@]}"; then break; fi
      echo "  submit failed (attempt $attempt), retrying in 20s..." >&2
      sleep 20
    done
  done
done
