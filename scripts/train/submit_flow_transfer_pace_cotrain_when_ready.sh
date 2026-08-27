#!/bin/bash
# Submit the two priority Flow Transfer cotrain arms on one PACE GPU each.
# Full jobs are allowed only after both exact-source smokes have semantic PASS.

set -Eeuo pipefail

PHASE=${PHASE:?set PHASE=smoke or PHASE=full}
GPU_TYPE=${GPU_TYPE:-a100}
EXPECTED_ENV_SHA=${EXPECTED_ENV_SHA:?set the exact PACE environment manifest SHA256}
LATENT_NORM_ARTIFACT=${LATENT_NORM_ARTIFACT:?exact horizon-100 norm_stats.json path}
LATENT_NORM_SHA=${LATENT_NORM_SHA:?exact horizon-100 norm SHA256}
DP_NORM_ARTIFACT=${DP_NORM_ARTIFACT:?exact horizon-16 norm_stats.json path}
DP_NORM_SHA=${DP_NORM_SHA:?exact horizon-16 norm SHA256}

REPO=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-flow-transfer-20260826
# This existing environment passed every trainHydra and config-target import;
# its complete package manifest is hashed into the smoke/full identity.
PY_ENV=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse/.venv
DATA_ROOT=/storage/project/r-dxu345-0/paphiwetsa3/datasets/flow_transfer_20260826
U_DATA=$DATA_ROOT/u_socket_3000_v2_clean
CHAIN_OBSTACLE_ROOT=$DATA_ROOT/chain_gripper_obstacle_3000_balanced_v1
EXP_ROOT=/storage/project/r-dxu345-0/paphiwetsa3/experiments/flow_transfer_direct_dense_obstacle3k_cotrain_pace_world1_normfix_20260826
LAUNCHER=$REPO/scripts/train/flow_transfer_direct_dense_matrix.sbatch
HELPER=$REPO/scripts/train/submit_flow_transfer_pace_cotrain_when_ready.sh
STATE_DIR=$EXP_ROOT/submission
SLURM_BIN=/opt/slurm/current/bin
ACCOUNT=gts-dxu345-rl2
QOS=inferno
export PATH=$SLURM_BIN:$PATH

case "$GPU_TYPE" in
  a100)
    GPU_MODEL=A100
    PARTITION=gpu-a100
    ;;
  h200)
    GPU_MODEL=H200
    PARTITION=gpu-h200
    ;;
  *)
    printf 'Unknown GPU_TYPE=%s; use a100 or h200\n' "$GPU_TYPE" >&2
    exit 64
    ;;
esac

case "$PHASE" in
  smoke)
    TIME_LIMIT=08:00:00
    OUTPUT_TSV=$STATE_DIR/smoke_jobs.tsv
    MARKER=$STATE_DIR/SMOKES_SUBMITTED
    ;;
  full)
    TIME_LIMIT=3-00:00:00
    OUTPUT_TSV=$STATE_DIR/full_jobs.tsv
    MARKER=$STATE_DIR/FULLS_SUBMITTED
    ;;
  *)
    printf 'Unknown PHASE=%s\n' "$PHASE" >&2
    exit 64
    ;;
esac

environment_manifest() {
  "$PY_ENV/bin/python" - <<'PY'
import importlib.metadata
import platform

print(f"python=={platform.python_version()}")
entries = set()
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if name:
        normalized = name.lower().replace("_", "-")
        entries.add(f"{normalized}=={distribution.version}")
for entry in sorted(entries):
    print(entry)
PY
}

EXPECTED_HEAD=$(git -C "$REPO" rev-parse HEAD)
EXPECTED_LAUNCHER_SHA=$(sha256sum "$LAUNCHER" | awk '{print $1}')
EXPECTED_HELPER_SHA=$(sha256sum "$HELPER" | awk '{print $1}')
test -x "$PY_ENV/bin/python"
test -x "$SLURM_BIN/sbatch"
test -x "$SLURM_BIN/sacct"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(environment_manifest | sha256sum | awk '{print $1}')" = "$EXPECTED_ENV_SHA"
test "$(sha256sum "$LATENT_NORM_ARTIFACT" | awk '{print $1}')" = "$LATENT_NORM_SHA"
test "$(sha256sum "$DP_NORM_ARTIFACT" | awk '{print $1}')" = "$DP_NORM_SHA"
test -s "$(dirname "$(dirname "$(dirname "$LATENT_NORM_ARTIFACT")")")/validation.json"
test -s "$(dirname "$(dirname "$(dirname "$DP_NORM_ARTIFACT")")")/validation.json"
test -f "$CHAIN_OBSTACLE_ROOT/audit_report.json"
test -f "$CHAIN_OBSTACLE_ROOT/subset_manifest.json"
test -f "$CHAIN_OBSTACLE_ROOT/inventory.txt"
test "$(find "$U_DATA" -mindepth 1 -maxdepth 1 -type d -name '*.zarr' | wc -l)" = 2999
test "$(find "$CHAIN_OBSTACLE_ROOT" -mindepth 4 -maxdepth 4 -type d -name '*.zarr' | wc -l)" = 3000

mkdir -p "$STATE_DIR" "$EXP_ROOT/slurm" "$EXP_ROOT/runs" \
  "$EXP_ROOT/smokes" "$EXP_ROOT/provenance"
LOCK_DIR=$STATE_DIR/.submit_${PHASE}_lock
if ! mkdir "$LOCK_DIR"; then
  printf 'Submission lock already exists: %s\n' "$LOCK_DIR" >&2
  exit 73
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
test ! -e "$MARKER"
test ! -e "$OUTPUT_TSV"
SMOKE_IDENTITY=$STATE_DIR/smoke_identity.tsv

if test "$PHASE" = smoke; then
  test ! -e "$SMOKE_IDENTITY"
  printf 'head\t%s\nlauncher_sha256\t%s\nhelper_sha256\t%s\nlatent_norm_artifact\t%s\nlatent_norm_sha256\t%s\ndp_norm_artifact\t%s\ndp_norm_sha256\t%s\nenvironment_sha256\t%s\ngpu_type\t%s\npartition\t%s\naccount\t%s\n' \
    "$EXPECTED_HEAD" "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_HELPER_SHA" \
    "$LATENT_NORM_ARTIFACT" "$LATENT_NORM_SHA" \
    "$DP_NORM_ARTIFACT" "$DP_NORM_SHA" "$EXPECTED_ENV_SHA" \
    "$GPU_TYPE" "$PARTITION" "$ACCOUNT" > "$SMOKE_IDENTITY"
else
  test -s "$SMOKE_IDENTITY"
  identity_value() {
    awk -F '\t' -v key="$1" '$1 == key {print $2}' "$SMOKE_IDENTITY"
  }
  test "$(identity_value head)" = "$EXPECTED_HEAD"
  test "$(identity_value launcher_sha256)" = "$EXPECTED_LAUNCHER_SHA"
  test "$(identity_value helper_sha256)" = "$EXPECTED_HELPER_SHA"
  test "$(identity_value latent_norm_artifact)" = "$LATENT_NORM_ARTIFACT"
  test "$(identity_value latent_norm_sha256)" = "$LATENT_NORM_SHA"
  test "$(identity_value dp_norm_artifact)" = "$DP_NORM_ARTIFACT"
  test "$(identity_value dp_norm_sha256)" = "$DP_NORM_SHA"
  test "$(identity_value environment_sha256)" = "$EXPECTED_ENV_SHA"
  test "$(identity_value gpu_type)" = "$GPU_TYPE"
  test "$(identity_value partition)" = "$PARTITION"
  test "$(identity_value account)" = "$ACCOUNT"
fi

"$SLURM_BIN/sbatch" --test-only \
  --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=64G \
  --time="$TIME_LIMIT" --gres="gpu:$GPU_TYPE:1" --wrap=true

ARMS=(cotrain_obstacle_latent cotrain_obstacle_dp)
if test "$PHASE" = full; then
  test -s "$STATE_DIR/smoke_jobs.tsv"
  "$PY_ENV/bin/python" - "$STATE_DIR/smoke_jobs.tsv" "$EXP_ROOT" \
    "$PARTITION" "$ACCOUNT" "$GPU_TYPE" <<'PY'
import csv
import json
import pathlib
import subprocess
import sys

jobs_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
partition, account, gpu_type = sys.argv[3:]
rows = list(csv.DictReader(jobs_path.open(), delimiter="\t"))
expected_arms = {"cotrain_obstacle_latent", "cotrain_obstacle_dp"}
assert {row["arm"] for row in rows} == expected_arms
for row in rows:
    arm = row["arm"]
    job = row["job_id"]
    assert int(row["gpus"]) == 1
    assert row["partition"] == partition
    assert row["account"] == account
    assert row["gpu_type"] == gpu_type
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
    assert validation["global_step"] == 3200
    assert validation["gradient_clipping_enabled"] is False
    assert set(validation["global_batch_per_domain"].values()) == {64}
    assert validation["total_global_batch"] == 128
PY
fi

: > "$OUTPUT_TSV"
printf 'arm\tjob_id\tpartition\taccount\tgpu_type\tgpus\tnorm_artifact\tnorm_sha256\n' >> "$OUTPUT_TSV"

for arm in "${ARMS[@]}"; do
  if [[ "$arm" == *_latent ]]; then
    norm_artifact=$LATENT_NORM_ARTIFACT
    norm_sha=$LATENT_NORM_SHA
  else
    norm_artifact=$DP_NORM_ARTIFACT
    norm_sha=$DP_NORM_SHA
  fi
  if test "$PHASE" = smoke; then
    job_prefix=ftsm_pace
  else
    job_prefix=ft_pace
  fi
  job_id=$(
    "$SLURM_BIN/sbatch" --parsable \
      --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
      --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=64G \
      --time="$TIME_LIMIT" --gres="gpu:$GPU_TYPE:1" \
      --requeue --signal=USR1@300 --open-mode=append \
      --job-name="${job_prefix}_${arm:0:16}" \
      --output="$EXP_ROOT/slurm/${PHASE}_${arm}_%j.out" \
      --error="$EXP_ROOT/slurm/${PHASE}_${arm}_%j.err" \
      --export="ALL,ARM=$arm,MODE=$PHASE,GPUS_EXPECTED=1,EXPECTED_HEAD=$EXPECTED_HEAD,EXPECTED_LAUNCHER_SHA=$EXPECTED_LAUNCHER_SHA,NORM_ARTIFACT=$norm_artifact,EXPECTED_NORM_SHA=$norm_sha,CLUSTER_PROFILE=pace_world1,FLOW_TRANSFER_REPO=$REPO,FLOW_TRANSFER_PY_ENV=$PY_ENV,FLOW_TRANSFER_EXP_ROOT=$EXP_ROOT,FLOW_TRANSFER_U_DATA=$U_DATA,CHAIN_OBSTACLE_ROOT=$CHAIN_OBSTACLE_ROOT,EXPECTED_ENV_SHA=$EXPECTED_ENV_SHA,PACE_GPU_MODEL=$GPU_MODEL,PACE_PARTITION=$PARTITION" \
      "$LAUNCHER"
  )
  job_id=${job_id%%;*}
  printf '%s\t%s\t%s\t%s\t%s\t1\t%s\t%s\n' \
    "$arm" "$job_id" "$PARTITION" "$ACCOUNT" "$GPU_TYPE" \
    "$norm_artifact" "$norm_sha" | tee -a "$OUTPUT_TSV"
done

date --iso-8601=seconds > "$MARKER"
printf '%s PACE cotrain matrix submitted; full jobs are never chained automatically.\n' "$PHASE"
