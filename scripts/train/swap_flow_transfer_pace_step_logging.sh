#!/bin/bash
# Smoke and promote a full-state PACE continuation with dense step logging.

set -Eeuo pipefail

PHASE=${PHASE:?set PHASE=resume_smoke or PHASE=resume_full}
LATENT_PARENT_JOB_ID=${LATENT_PARENT_JOB_ID:?set the live latent predecessor job ID}
DP_PARENT_JOB_ID=${DP_PARENT_JOB_ID:?set the live DP predecessor job ID}
FULL_TARGET_ARM=${FULL_TARGET_ARM:-}

REPO=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-flow-transfer-step-logging-pace-20260827
PY_ENV=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse/.venv
DATA_ROOT=/storage/project/r-dxu345-0/paphiwetsa3/datasets/flow_transfer_20260826
U_DATA=$DATA_ROOT/u_socket_3000_v2_clean
CHAIN_OBSTACLE_ROOT=$DATA_ROOT/chain_gripper_obstacle_3000_balanced_v1
EXP_ROOT=/storage/project/r-dxu345-0/paphiwetsa3/experiments/flow_transfer_direct_dense_obstacle3k_cotrain_pace_h200_world1_normfix_20260826
LAUNCHER=$REPO/scripts/train/flow_transfer_direct_dense_matrix.sbatch
HELPER=$REPO/scripts/train/swap_flow_transfer_pace_step_logging.sh
STATE_DIR=$EXP_ROOT/submission
HANDOFF_ROOT=$EXP_ROOT/resume_handoffs
OLD_FULL_JOBS=$STATE_DIR/full_jobs.tsv
OLD_SMOKE_IDENTITY=$STATE_DIR/smoke_identity.tsv
IDENTITY=$STATE_DIR/step_logging_resume_smoke_identity.tsv
SMOKE_JOBS=$STATE_DIR/step_logging_resume_smoke_jobs.tsv
SLURM_BIN=/opt/slurm/current/bin
SRUN=$SLURM_BIN/srun
ACCOUNT=gts-dxu345-rl2
PARTITION=gpu-h200
GPU_TYPE=h200
GPU_MODEL=H200
QOS=inferno
SMOKE_TIME=00:30:00
ORIGINAL_FULL_SECONDS=172800
FULL_SAFETY_SECONDS=300

export PATH=$SLURM_BIN:$PATH

case "$PHASE" in
  resume_smoke)
    test -z "$FULL_TARGET_ARM"
    ARMS=(cotrain_obstacle_latent cotrain_obstacle_dp)
    OUTPUT_TSV=$SMOKE_JOBS
    MARKER=$STATE_DIR/STEP_LOGGING_RESUME_SMOKES_SUBMITTED
    LOCK_SUFFIX=resume_smoke
    AUDIT_PARENT_JOB_ID=$LATENT_PARENT_JOB_ID
    ;;
  resume_full)
    case "$FULL_TARGET_ARM" in
      cotrain_obstacle_latent)
        ARM_SUFFIX=latent
        AUDIT_PARENT_JOB_ID=$LATENT_PARENT_JOB_ID
        ;;
      cotrain_obstacle_dp)
        ARM_SUFFIX=dp
        AUDIT_PARENT_JOB_ID=$DP_PARENT_JOB_ID
        ;;
      *)
        printf 'Set FULL_TARGET_ARM to one cotrain arm for credit-safe staging.\n' >&2
        exit 64
        ;;
    esac
    ARMS=("$FULL_TARGET_ARM")
    OUTPUT_TSV=$STATE_DIR/step_logging_resume_full_${ARM_SUFFIX}_job.tsv
    MARKER=$STATE_DIR/STEP_LOGGING_RESUME_FULL_${ARM_SUFFIX^^}_SUBMITTED
    LOCK_SUFFIX=resume_full_$ARM_SUFFIX
    ;;
  *)
    printf 'Unknown PHASE=%s\n' "$PHASE" >&2
    exit 64
    ;;
esac

EXPECTED_HEAD=$(git -C "$REPO" rev-parse HEAD)
EXPECTED_LAUNCHER_SHA=$(sha256sum "$LAUNCHER" | awk '{print $1}')
EXPECTED_HELPER_SHA=$(sha256sum "$HELPER" | awk '{print $1}')
EXPECTED_ENV_SHA=$(awk -F '\t' '$1 == "environment_sha256" {print $2}' "$OLD_SMOKE_IDENTITY")

test -x "$PY_ENV/bin/python"
test -x "$SLURM_BIN/sbatch"
test -x "$SLURM_BIN/sacct"
test -x "$SLURM_BIN/squeue"
test -x "$SLURM_BIN/scancel"
test -x "$SRUN"
test -s "$OLD_FULL_JOBS"
test -s "$OLD_SMOKE_IDENTITY"
test -n "$EXPECTED_ENV_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test ! -e "$MARKER"
test ! -e "$OUTPUT_TSV"
mkdir -p "$STATE_DIR" "$HANDOFF_ROOT" "$EXP_ROOT/slurm"

LOCK_DIR=$STATE_DIR/.step_logging_${LOCK_SUFFIX}_lock
if ! mkdir "$LOCK_DIR"; then
  printf 'Submission lock already exists: %s\n' "$LOCK_DIR" >&2
  exit 73
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

old_row() {
  local arm=$1 field=$2
  awk -F '\t' -v arm="$arm" -v field="$field" '
    NR == 1 {
      for (i = 1; i <= NF; ++i) column[$i] = i
      next
    }
    $1 == arm {print $(column[field])}
  ' "$OLD_FULL_JOBS"
}

arm_selected() {
  local sought=$1 selected
  for selected in "${ARMS[@]}"; do
    if test "$selected" = "$sought"; then
      return 0
    fi
  done
  return 1
}

seconds_to_slurm() {
  local total=$1
  local days hours minutes seconds
  days=$((total / 86400))
  hours=$(((total % 86400) / 3600))
  minutes=$(((total % 3600) / 60))
  seconds=$((total % 60))
  printf '%d-%02d:%02d:%02d\n' "$days" "$hours" "$minutes" "$seconds"
}

assert_live_parent() {
  local job=$1 expected_arm=$2
  local state
  state=$("$SLURM_BIN/squeue" -h -j "$job" -o '%T')
  test "$state" = RUNNING
  test "$(old_row "$expected_arm" job_id)" = "$job"
}

snapshot_checkpoint() {
  local arm=$1 parent=$2 tag=$3
  local run_dir=$EXP_ROOT/runs/${arm}_seed42
  local source=$run_dir/checkpoints/last.ckpt
  local destination_dir=$HANDOFF_ROOT/$arm/from_job_$parent/$tag
  local before after temp metadata step epoch scheduler_step destination sha manifest

  test -s "$source"
  mkdir -p "$destination_dir"
  before=$(stat -c '%s:%y' "$source")
  temp=$destination_dir/.last.ckpt.copying
  test ! -e "$temp"
  "$SRUN" --jobid="$parent" --overlap --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --quiet \
    cp --reflink=auto --preserve=timestamps "$source" "$temp"
  after=$(stat -c '%s:%y' "$source")
  test "$before" = "$after"
  metadata=$(
    "$SRUN" --jobid="$parent" --overlap --nodes=1 --ntasks=1 \
      --cpus-per-task=1 --quiet /usr/bin/env CUDA_VISIBLE_DEVICES= \
      "$PY_ENV/bin/python" - "$temp" <<'PY'
import pathlib
import sys

import torch

path = pathlib.Path(sys.argv[1])
checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
step = int(checkpoint["global_step"])
epoch = int(checkpoint["epoch"])
assert checkpoint.get("state_dict")
assert checkpoint.get("optimizer_states")
assert checkpoint.get("lr_schedulers")
scheduler_step = int(checkpoint["lr_schedulers"][0]["last_epoch"])
assert scheduler_step == step
print(f"{step}\t{epoch}\t{scheduler_step}")
PY
  )
  IFS=$'\t' read -r step epoch scheduler_step <<< "$metadata"
  destination=$destination_dir/step_${step}.ckpt
  test ! -e "$destination"
  mv "$temp" "$destination"
  chmod 444 "$destination"
  sha=$(
    "$SRUN" --jobid="$parent" --overlap --nodes=1 --ntasks=1 \
      --cpus-per-task=1 --quiet sha256sum "$destination" | awk '{print $1}'
  )
  manifest=$destination_dir/step_${step}.json
  "$SRUN" --jobid="$parent" --overlap --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --quiet /usr/bin/env CUDA_VISIBLE_DEVICES= \
    "$PY_ENV/bin/python" - "$manifest" "$arm" "$parent" "$destination" \
      "$sha" "$step" "$epoch" "$scheduler_step" "$before" "$EXPECTED_HEAD" <<'PY'
import json
import pathlib
import sys

(
    output,
    arm,
    parent,
    checkpoint,
    sha,
    step,
    epoch,
    scheduler_step,
    source_stat,
    source_head,
) = sys.argv[1:]
payload = {
    "arm": arm,
    "parent_job_id": int(parent),
    "checkpoint": checkpoint,
    "checkpoint_sha256": sha,
    "global_step": int(step),
    "epoch": int(epoch),
    "scheduler_last_epoch": int(scheduler_step),
    "source_stat": source_stat,
    "continuation_source_head": source_head,
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  chmod 444 "$manifest"
  printf '%s\t%s\t%s\t%s\t%s\n' "$destination" "$sha" "$step" "$epoch" "$manifest"
}

if arm_selected cotrain_obstacle_latent; then
  assert_live_parent "$LATENT_PARENT_JOB_ID" cotrain_obstacle_latent
fi
if arm_selected cotrain_obstacle_dp; then
  assert_live_parent "$DP_PARENT_JOB_ID" cotrain_obstacle_dp
fi

if test "$PHASE" = resume_full; then
  test -s "$IDENTITY"
  test -s "$SMOKE_JOBS"
  identity_value() {
    awk -F '\t' -v key="$1" '$1 == key {print $2}' "$IDENTITY"
  }
  test "$(identity_value head)" = "$EXPECTED_HEAD"
  test "$(identity_value launcher_sha256)" = "$EXPECTED_LAUNCHER_SHA"
  test "$(identity_value helper_sha256)" = "$EXPECTED_HELPER_SHA"
  "$SRUN" --jobid="$AUDIT_PARENT_JOB_ID" --overlap --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --quiet /usr/bin/env CUDA_VISIBLE_DEVICES= \
    "$PY_ENV/bin/python" - "$SMOKE_JOBS" "$EXP_ROOT" <<'PY'
import csv
import json
import pathlib
import subprocess
import sys

jobs_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
rows = list(csv.DictReader(jobs_path.open(), delimiter="\t"))
assert {row["arm"] for row in rows} == {
    "cotrain_obstacle_latent",
    "cotrain_obstacle_dp",
}
for row in rows:
    job = row["job_id"]
    accounting = subprocess.check_output(
        ["/opt/slurm/current/bin/sacct", "-nX", "-j", job, "-o", "State,ExitCode"],
        text=True,
    ).strip().split()
    assert accounting[:2] == ["COMPLETED", "0:0"], (job, accounting)
    validation_path = (
        root
        / "provenance"
        / "resume_smoke"
        / row["arm"]
        / f"job_{job}"
        / "run_validation.json"
    )
    result = json.loads(validation_path.read_text())
    assert result["status"] == "PASS"
    assert result["mode"] == "resume_smoke"
    assert result["resumed_steps"] == 1
    assert result["validation_enabled"] is True
    assert result["resume_contract_match"] is True
    assert result["scheduler_last_epoch"] == result["global_step"]
    assert result["resume_checkpoint_sha256"] == row["checkpoint_sha256"]
PY
fi

latent_norm=$(old_row cotrain_obstacle_latent norm_artifact)
latent_norm_sha=$(old_row cotrain_obstacle_latent norm_sha256)
dp_norm=$(old_row cotrain_obstacle_dp norm_artifact)
dp_norm_sha=$(old_row cotrain_obstacle_dp norm_sha256)
test "$(sha256sum "$latent_norm" | awk '{print $1}')" = "$latent_norm_sha"
test "$(sha256sum "$dp_norm" | awk '{print $1}')" = "$dp_norm_sha"

snapshot_tag=${PHASE}_$(date +%Y%m%d_%H%M%S)
latent_ckpt=''
latent_ckpt_sha=''
latent_step=''
latent_epoch=''
latent_manifest=''
dp_ckpt=''
dp_ckpt_sha=''
dp_step=''
dp_epoch=''
dp_manifest=''
if arm_selected cotrain_obstacle_latent; then
  IFS=$'\t' read -r latent_ckpt latent_ckpt_sha latent_step latent_epoch latent_manifest < <(
    snapshot_checkpoint cotrain_obstacle_latent "$LATENT_PARENT_JOB_ID" "$snapshot_tag"
  )
fi
if arm_selected cotrain_obstacle_dp; then
  IFS=$'\t' read -r dp_ckpt dp_ckpt_sha dp_step dp_epoch dp_manifest < <(
    snapshot_checkpoint cotrain_obstacle_dp "$DP_PARENT_JOB_ID" "$snapshot_tag"
  )
fi

if test "$PHASE" = resume_full; then
  if arm_selected cotrain_obstacle_latent; then
    test "$latent_ckpt_sha" != "$(identity_value latent_checkpoint_sha256)"
  fi
  if arm_selected cotrain_obstacle_dp; then
    test "$dp_ckpt_sha" != "$(identity_value dp_checkpoint_sha256)"
  fi
fi

if test "$PHASE" = resume_smoke; then
  "$SLURM_BIN/sbatch" --test-only \
    --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=64G \
    --time="$SMOKE_TIME" --gres="gpu:$GPU_TYPE:1" --no-requeue --wrap=true
fi

if test "$PHASE" = resume_smoke; then
  printf 'head\t%s\nlauncher_sha256\t%s\nhelper_sha256\t%s\nenvironment_sha256\t%s\nlatent_parent_job_id\t%s\nlatent_checkpoint\t%s\nlatent_checkpoint_sha256\t%s\nlatent_checkpoint_step\t%s\nlatent_checkpoint_epoch\t%s\nlatent_checkpoint_manifest\t%s\ndp_parent_job_id\t%s\ndp_checkpoint\t%s\ndp_checkpoint_sha256\t%s\ndp_checkpoint_step\t%s\ndp_checkpoint_epoch\t%s\ndp_checkpoint_manifest\t%s\n' \
    "$EXPECTED_HEAD" "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_HELPER_SHA" \
    "$EXPECTED_ENV_SHA" "$LATENT_PARENT_JOB_ID" "$latent_ckpt" \
    "$latent_ckpt_sha" "$latent_step" "$latent_epoch" "$latent_manifest" \
    "$DP_PARENT_JOB_ID" "$dp_ckpt" "$dp_ckpt_sha" "$dp_step" \
    "$dp_epoch" "$dp_manifest" > "$IDENTITY"
fi

printf 'arm\tjob_id\tparent_job_id\tcheckpoint\tcheckpoint_sha256\tcheckpoint_manifest\tresume_step\tresume_epoch\tpartition\taccount\tgpu_type\ttime_limit\tdependency\n' \
  > "$OUTPUT_TSV"

for arm in "${ARMS[@]}"; do
  if test "$arm" = cotrain_obstacle_latent; then
    parent=$LATENT_PARENT_JOB_ID
    checkpoint=$latent_ckpt
    checkpoint_sha=$latent_ckpt_sha
    checkpoint_manifest=$latent_manifest
    resume_step=$latent_step
    resume_epoch=$latent_epoch
    norm_artifact=$latent_norm
    norm_sha=$latent_norm_sha
  else
    parent=$DP_PARENT_JOB_ID
    checkpoint=$dp_ckpt
    checkpoint_sha=$dp_ckpt_sha
    checkpoint_manifest=$dp_manifest
    resume_step=$dp_step
    resume_epoch=$dp_epoch
    norm_artifact=$dp_norm
    norm_sha=$dp_norm_sha
  fi

  dependency=none
  time_limit=$SMOKE_TIME
  dependency_args=()
  if test "$PHASE" = resume_full; then
    elapsed_raw=$("$SLURM_BIN/sacct" -nX -j "$parent" -o ElapsedRaw -P | sed -n '1{s/|.*//;p;}')
    test "$elapsed_raw" -eq "$elapsed_raw"
    remaining=$((ORIGINAL_FULL_SECONDS - elapsed_raw - FULL_SAFETY_SECONDS))
    test "$remaining" -gt 3600
    time_limit=$(seconds_to_slurm "$remaining")
    dependency=afterany:$parent
    dependency_args=(--dependency="$dependency")
  fi

  "$SLURM_BIN/sbatch" --test-only \
    --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=64G \
    --time="$time_limit" --gres="gpu:$GPU_TYPE:1" --no-requeue \
    "${dependency_args[@]}" --wrap=true

  job_id=$(
    "$SLURM_BIN/sbatch" --parsable \
      --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
      --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=64G \
      --time="$time_limit" --gres="gpu:$GPU_TYPE:1" --no-requeue \
      "${dependency_args[@]}" --open-mode=append \
      --job-name="ftsl_${PHASE}_${arm:17:6}" \
      --output="$EXP_ROOT/slurm/${PHASE}_${arm}_%j.out" \
      --error="$EXP_ROOT/slurm/${PHASE}_${arm}_%j.err" \
      --export="ALL,ARM=$arm,MODE=$PHASE,GPUS_EXPECTED=1,EXPECTED_HEAD=$EXPECTED_HEAD,EXPECTED_LAUNCHER_SHA=$EXPECTED_LAUNCHER_SHA,NORM_ARTIFACT=$norm_artifact,EXPECTED_NORM_SHA=$norm_sha,CLUSTER_PROFILE=pace_world1,FLOW_TRANSFER_REPO=$REPO,FLOW_TRANSFER_PY_ENV=$PY_ENV,FLOW_TRANSFER_EXP_ROOT=$EXP_ROOT,FLOW_TRANSFER_U_DATA=$U_DATA,CHAIN_OBSTACLE_ROOT=$CHAIN_OBSTACLE_ROOT,EXPECTED_ENV_SHA=$EXPECTED_ENV_SHA,PACE_GPU_MODEL=$GPU_MODEL,PACE_PARTITION=$PARTITION,EXPLICIT_RESUME_CKPT=$checkpoint,EXPECTED_RESUME_CKPT_SHA=$checkpoint_sha,EXPECTED_RESUME_GLOBAL_STEP=$resume_step,RESUME_PARENT_JOB_ID=$parent" \
      "$LAUNCHER"
  )
  job_id=${job_id%%;*}
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$job_id" "$parent" "$checkpoint" "$checkpoint_sha" \
    "$checkpoint_manifest" "$resume_step" "$resume_epoch" "$PARTITION" \
    "$ACCOUNT" "$GPU_TYPE" "$time_limit" "$dependency" | tee -a "$OUTPUT_TSV"
done

date --iso-8601=seconds > "$MARKER"
if test "$PHASE" = resume_full; then
  printf 'One continuation is dependency-gated. Audit it before cancelling parent %s.\n' \
    "$AUDIT_PARENT_JOB_ID"
else
  printf 'Exact full-state resume smokes submitted; do not replace predecessors until both PASS.\n'
fi
