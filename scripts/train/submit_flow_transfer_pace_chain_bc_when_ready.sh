#!/bin/bash
# Submit only the two obstacle-only ChainGripper BC arms on one PACE A100 each.
# The static worst-case charge covers both smokes and both full allocations;
# full jobs are allowed only after both exact-source smokes have semantic PASS.

set -Eeuo pipefail

PHASE=${PHASE:?set PHASE=smoke or PHASE=full}
EXPECTED_ENV_SHA=${EXPECTED_ENV_SHA:?set the exact PACE environment manifest SHA256}
LATENT_NORM_ARTIFACT=${LATENT_NORM_ARTIFACT:?exact horizon-100 norm_stats.json path}
LATENT_NORM_SHA=${LATENT_NORM_SHA:?exact horizon-100 norm SHA256}
DP_NORM_ARTIFACT=${DP_NORM_ARTIFACT:?exact horizon-16 norm_stats.json path}
DP_NORM_SHA=${DP_NORM_SHA:?exact horizon-16 norm SHA256}

REPO=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-flow-transfer-chain-bc-pace-20260827
PY_ENV=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse/.venv
DATA_ROOT=/storage/project/r-dxu345-0/paphiwetsa3/datasets/flow_transfer_20260826
U_DATA=$DATA_ROOT/u_socket_3000_v2_clean
CHAIN_OBSTACLE_ROOT=$DATA_ROOT/chain_gripper_obstacle_3000_balanced_v1
EXP_ROOT=/storage/project/r-dxu345-0/paphiwetsa3/experiments/flow_transfer_chain_bc_obstacle3k_pace_a100_world1_normfix_20260827
LAUNCHER=$REPO/scripts/train/flow_transfer_direct_dense_matrix.sbatch
HELPER=$REPO/scripts/train/submit_flow_transfer_pace_chain_bc_when_ready.sh
STATE_DIR=$EXP_ROOT/submission
SLURM_BIN=/opt/slurm/current/bin
ACCOUNT=gts-dxu345-rl2
QOS=inferno
PARTITION=gpu-a100
GPU_TYPE=a100
GPU_MODEL=A100
GPU_CONSTRAINT=A100-80GB
SMOKE_HOURS=2
FULL_HOURS=30
A100_CREDITS_PER_GPU_HOUR=0.276884
BC_MAX_TOTAL_CREDITS=20.0
export PATH=$SLURM_BIN:$PATH

case "$PHASE" in
  smoke)
    TIME_LIMIT=02:00:00
    OUTPUT_TSV=$STATE_DIR/smoke_jobs.tsv
    MARKER=$STATE_DIR/SMOKES_SUBMITTED
    ;;
  full)
    TIME_LIMIT=1-06:00:00
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
test "$(find "$CHAIN_OBSTACLE_ROOT" -mindepth 4 -maxdepth 4 -type d -name '*.zarr' | wc -l)" = 3000

EXPECTED_MAX_CREDITS=$(
  "$PY_ENV/bin/python" - "$SMOKE_HOURS" "$FULL_HOURS" \
    "$A100_CREDITS_PER_GPU_HOUR" "$BC_MAX_TOTAL_CREDITS" <<'PY'
import sys

smoke_hours, full_hours, rate, cap = map(float, sys.argv[1:])
cost = 2 * (smoke_hours + full_hours) * rate
assert cost <= cap, (cost, cap)
print(f"{cost:.6f}")
PY
)

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
  printf 'head\t%s\nlauncher_sha256\t%s\nhelper_sha256\t%s\nlatent_norm_artifact\t%s\nlatent_norm_sha256\t%s\ndp_norm_artifact\t%s\ndp_norm_sha256\t%s\nenvironment_sha256\t%s\ngpu_type\t%s\npartition\t%s\ngpu_constraint\t%s\naccount\t%s\nsmoke_hours_per_arm\t%s\nfull_hours_per_arm\t%s\ncredit_rate\t%s\nmaximum_total_credits\t%s\ncredit_cap\t%s\n' \
    "$EXPECTED_HEAD" "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_HELPER_SHA" \
    "$LATENT_NORM_ARTIFACT" "$LATENT_NORM_SHA" \
    "$DP_NORM_ARTIFACT" "$DP_NORM_SHA" "$EXPECTED_ENV_SHA" \
    "$GPU_TYPE" "$PARTITION" "$GPU_CONSTRAINT" "$ACCOUNT" \
    "$SMOKE_HOURS" "$FULL_HOURS" "$A100_CREDITS_PER_GPU_HOUR" \
    "$EXPECTED_MAX_CREDITS" "$BC_MAX_TOTAL_CREDITS" > "$SMOKE_IDENTITY"
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
  test "$(identity_value gpu_constraint)" = "$GPU_CONSTRAINT"
  test "$(identity_value account)" = "$ACCOUNT"
  test "$(identity_value smoke_hours_per_arm)" = "$SMOKE_HOURS"
  test "$(identity_value full_hours_per_arm)" = "$FULL_HOURS"
  test "$(identity_value credit_rate)" = "$A100_CREDITS_PER_GPU_HOUR"
  test "$(identity_value maximum_total_credits)" = "$EXPECTED_MAX_CREDITS"
  test "$(identity_value credit_cap)" = "$BC_MAX_TOTAL_CREDITS"
fi

"$SLURM_BIN/sbatch" --test-only \
  --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=96G \
  --time="$TIME_LIMIT" --gres="gpu:$GPU_TYPE:1" \
  --constraint="$GPU_CONSTRAINT" --wrap=true

ARMS=(bc_chain_latent bc_chain_dp)
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
expected_arms = {"bc_chain_latent", "bc_chain_dp"}
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
    assert validation["total_global_batch"] == 64
    assert validation["validation_enabled"] is True
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
    job_prefix=ftsm_bc
  else
    job_prefix=ft_bc
  fi
  job_id=$(
    "$SLURM_BIN/sbatch" --parsable \
      --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
      --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=96G \
      --time="$TIME_LIMIT" --gres="gpu:$GPU_TYPE:1" \
      --constraint="$GPU_CONSTRAINT" --no-requeue \
      --open-mode=append \
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
printf '%s PACE ChainGripper BC pair submitted; maximum cohort charge is %s credits, and full jobs are never chained automatically.\n' \
  "$PHASE" "$EXPECTED_MAX_CREDITS"
