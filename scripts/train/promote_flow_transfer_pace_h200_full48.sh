#!/bin/bash
# Promote the exact c3c6da85 H200 cotrain smokes to hard-capped 48-hour full
# jobs.  This controller is deliberately separate from the immutable training
# worktree used by the smokes, so adding the cap cannot invalidate their source
# identity.

set -Eeuo pipefail

TRAIN_REPO=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-flow-transfer-h200-20260826
PY_ENV=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse/.venv
DATA_ROOT=/storage/project/r-dxu345-0/paphiwetsa3/datasets/flow_transfer_20260826
U_DATA=$DATA_ROOT/u_socket_3000_v2_clean
CHAIN_OBSTACLE_ROOT=$DATA_ROOT/chain_gripper_obstacle_3000_balanced_v1
EXP_ROOT=/storage/project/r-dxu345-0/paphiwetsa3/experiments/flow_transfer_direct_dense_obstacle3k_cotrain_pace_h200_world1_normfix_20260826
LAUNCHER=$TRAIN_REPO/scripts/train/flow_transfer_direct_dense_matrix.sbatch
SMOKE_HELPER=$TRAIN_REPO/scripts/train/submit_flow_transfer_pace_cotrain_when_ready.sh
STATE_DIR=$EXP_ROOT/submission
SLURM_BIN=/opt/slurm/current/bin
ACCOUNT=gts-dxu345-rl2
PARTITION=gpu-h200
GPU_TYPE=h200
GPU_MODEL=H200
QOS=inferno
TIME_LIMIT=2-00:00:00

EXPECTED_TRAIN_HEAD=c3c6da8576212b16a31724ab6d1826b9513f51c7
EXPECTED_LAUNCHER_SHA=2ab20126bb89060a9f438a4b60392390c52a7bc7a96cfb42561068a67ce96912
EXPECTED_SMOKE_HELPER_SHA=b3f35ae2477fd0028fe6e3e216d2d911f4d7b3af5fb9b762ae62a23fb4a36dcd

SCRIPT_PATH=$(readlink -f "$0")
CONTROL_REPO=$(git -C "$(dirname "$SCRIPT_PATH")" rev-parse --show-toplevel)
CONTROL_HEAD=$(git -C "$CONTROL_REPO" rev-parse HEAD)
CONTROL_SHA=$(sha256sum "$SCRIPT_PATH" | awk '{print $1}')
SMOKE_IDENTITY=$STATE_DIR/smoke_identity.tsv
SMOKE_JOBS=$STATE_DIR/smoke_jobs.tsv
FULL_JOBS=$STATE_DIR/full_jobs.tsv
FULL_MARKER=$STATE_DIR/FULLS_SUBMITTED
CONTROL_IDENTITY=$STATE_DIR/full48_controller_identity.tsv
LOCK_DIR=$STATE_DIR/.submit_full48_lock

export PATH=$SLURM_BIN:$PATH

identity_value() {
  awk -F '\t' -v key="$1" '$1 == key {print $2}' "$SMOKE_IDENTITY"
}

test -x "$PY_ENV/bin/python"
test -x "$SLURM_BIN/sbatch"
test -x "$SLURM_BIN/sacct"
test -s "$SMOKE_IDENTITY"
test -s "$SMOKE_JOBS"
test ! -e "$FULL_MARKER"
test ! -e "$FULL_JOBS"
test ! -e "$CONTROL_IDENTITY"
test "$(git -C "$TRAIN_REPO" rev-parse HEAD)" = "$EXPECTED_TRAIN_HEAD"
test -z "$(git -C "$TRAIN_REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
test "$(sha256sum "$SMOKE_HELPER" | awk '{print $1}')" = "$EXPECTED_SMOKE_HELPER_SHA"
test -z "$(git -C "$CONTROL_REPO" status --porcelain=v1 --untracked-files=all)"

test "$(identity_value head)" = "$EXPECTED_TRAIN_HEAD"
test "$(identity_value launcher_sha256)" = "$EXPECTED_LAUNCHER_SHA"
test "$(identity_value helper_sha256)" = "$EXPECTED_SMOKE_HELPER_SHA"
test "$(identity_value gpu_type)" = "$GPU_TYPE"
test "$(identity_value partition)" = "$PARTITION"
test "$(identity_value gpu_constraint)" = ""
test "$(identity_value account)" = "$ACCOUNT"

EXPECTED_ENV_SHA=$(identity_value environment_sha256)
LATENT_NORM_ARTIFACT=$(identity_value latent_norm_artifact)
LATENT_NORM_SHA=$(identity_value latent_norm_sha256)
DP_NORM_ARTIFACT=$(identity_value dp_norm_artifact)
DP_NORM_SHA=$(identity_value dp_norm_sha256)
test -n "$EXPECTED_ENV_SHA"
test "$(sha256sum "$LATENT_NORM_ARTIFACT" | awk '{print $1}')" = "$LATENT_NORM_SHA"
test "$(sha256sum "$DP_NORM_ARTIFACT" | awk '{print $1}')" = "$DP_NORM_SHA"

if ! mkdir "$LOCK_DIR"; then
  printf 'Submission lock already exists: %s\n' "$LOCK_DIR" >&2
  exit 73
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

"$PY_ENV/bin/python" - "$SMOKE_JOBS" "$EXP_ROOT" <<'PY'
import csv
import json
import pathlib
import subprocess
import sys

jobs_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
rows = list(csv.DictReader(jobs_path.open(), delimiter="\t"))
expected = {"cotrain_obstacle_latent", "cotrain_obstacle_dp"}
assert {row["arm"] for row in rows} == expected
for row in rows:
    arm = row["arm"]
    job = row["job_id"]
    assert int(row["gpus"]) == 1
    assert row["partition"] == "gpu-h200"
    assert row["account"] == "gts-dxu345-rl2"
    assert row["gpu_type"] == "h200"
    accounting = subprocess.check_output(
        [
            "/opt/slurm/current/bin/sacct",
            "-nX",
            "-j",
            job,
            "-o",
            "State,ExitCode",
        ],
        text=True,
    ).strip().split()
    assert accounting[:2] == ["COMPLETED", "0:0"], (job, accounting)
    validation_path = (
        root / "provenance" / "smoke" / arm / f"job_{job}" / "run_validation.json"
    )
    validation = json.loads(validation_path.read_text())
    assert validation["status"] == "PASS"
    assert validation["mode"] == "smoke"
    assert validation["expected_world_size"] == 1
    assert validation["global_step"] == 3_200
    assert validation["gradient_clipping_enabled"] is False
    assert set(validation["global_batch_per_domain"].values()) == {64}
    assert validation["total_global_batch"] == 128
PY

# Test the exact request shape before creating any state or submitting a job.
"$SLURM_BIN/sbatch" --test-only \
  --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=64G \
  --time="$TIME_LIMIT" --gres="gpu:$GPU_TYPE:1" --no-requeue --wrap=true

printf 'controller_head\t%s\ncontroller_sha256\t%s\ntrain_head\t%s\nlauncher_sha256\t%s\nsmoke_helper_sha256\t%s\ntime_limit\t%s\nrequeue\tfalse\nsignal\tnone\n' \
  "$CONTROL_HEAD" "$CONTROL_SHA" "$EXPECTED_TRAIN_HEAD" \
  "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_SMOKE_HELPER_SHA" "$TIME_LIMIT" \
  > "$CONTROL_IDENTITY"

printf 'arm\tjob_id\tpartition\taccount\tgpu_type\tgpus\tnorm_artifact\tnorm_sha256\ttime_limit\trequeue\n' \
  > "$FULL_JOBS"

for arm in cotrain_obstacle_latent cotrain_obstacle_dp; do
  if [[ "$arm" == *_latent ]]; then
    norm_artifact=$LATENT_NORM_ARTIFACT
    norm_sha=$LATENT_NORM_SHA
  else
    norm_artifact=$DP_NORM_ARTIFACT
    norm_sha=$DP_NORM_SHA
  fi
  job_id=$(
    "$SLURM_BIN/sbatch" --parsable \
      --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
      --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=64G \
      --time="$TIME_LIMIT" --gres="gpu:$GPU_TYPE:1" --no-requeue \
      --open-mode=append \
      --job-name="ft_pace_${arm:0:16}" \
      --output="$EXP_ROOT/slurm/full_${arm}_%j.out" \
      --error="$EXP_ROOT/slurm/full_${arm}_%j.err" \
      --export="ALL,ARM=$arm,MODE=full,GPUS_EXPECTED=1,EXPECTED_HEAD=$EXPECTED_TRAIN_HEAD,EXPECTED_LAUNCHER_SHA=$EXPECTED_LAUNCHER_SHA,NORM_ARTIFACT=$norm_artifact,EXPECTED_NORM_SHA=$norm_sha,CLUSTER_PROFILE=pace_world1,FLOW_TRANSFER_REPO=$TRAIN_REPO,FLOW_TRANSFER_PY_ENV=$PY_ENV,FLOW_TRANSFER_EXP_ROOT=$EXP_ROOT,FLOW_TRANSFER_U_DATA=$U_DATA,CHAIN_OBSTACLE_ROOT=$CHAIN_OBSTACLE_ROOT,EXPECTED_ENV_SHA=$EXPECTED_ENV_SHA,PACE_GPU_MODEL=$GPU_MODEL,PACE_PARTITION=$PARTITION" \
      "$LAUNCHER"
  )
  job_id=${job_id%%;*}
  printf '%s\t%s\t%s\t%s\t%s\t1\t%s\t%s\t%s\tfalse\n' \
    "$arm" "$job_id" "$PARTITION" "$ACCOUNT" "$GPU_TYPE" \
    "$norm_artifact" "$norm_sha" "$TIME_LIMIT" | tee -a "$FULL_JOBS"
done

date --iso-8601=seconds > "$FULL_MARKER"
printf 'Two exact-source H200 cotrain full jobs submitted with a hard aggregate 48-hour cap each.\n'
