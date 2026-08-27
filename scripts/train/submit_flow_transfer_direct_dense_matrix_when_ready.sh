#!/bin/bash
# Submit either the corrected six-arm matrix or the four BC-only arms. Full
# jobs are allowed only after every selected smoke has a semantic PASS.
# Normalization paths and SHAs are mandatory.

set -Eeuo pipefail

PHASE=${PHASE:?set PHASE=smoke or PHASE=full}
MATRIX_SCOPE=${MATRIX_SCOPE:-bc}
LATENT_NORM_ARTIFACT=${LATENT_NORM_ARTIFACT:?exact H100 norm_stats.json path}
LATENT_NORM_SHA=${LATENT_NORM_SHA:?exact H100 norm SHA256}
DP_NORM_ARTIFACT=${DP_NORM_ARTIFACT:?exact H16 norm_stats.json path}
DP_NORM_SHA=${DP_NORM_SHA:?exact H16 norm SHA256}
EXPECTED_ENV_SHA=8fd1504c955756adf8167f7bd34fc1a09cb844d268898b3ac30693e99ac60e87

REPO=${FLOW_TRANSFER_REPO:-/coc/flash7/paphiwetsa3/worktrees/flow-transfer-bc-skynet-20260827}
LAUNCHER=$REPO/scripts/train/flow_transfer_direct_dense_matrix.sbatch
case "$MATRIX_SCOPE" in
  all)
    EXP_ROOT=${FLOW_TRANSFER_EXP_ROOT:-/coc/flash7/paphiwetsa3/experiments/flow_transfer_direct_dense_obstacle3k_cotrain_world2_normfix_20260826}
    ARMS=(
      bc_usocket_latent
      bc_usocket_dp
      bc_chain_latent
      bc_chain_dp
      cotrain_obstacle_latent
      cotrain_obstacle_dp
    )
    RESOURCE_SPECS=(
      'hoffman-lab hoffman-lab a40 1 96G'
      'rl2-lab rl2-lab l40s 2 128G'
    )
    ;;
  bc)
    EXP_ROOT=${FLOW_TRANSFER_EXP_ROOT:-/coc/flash7/paphiwetsa3/experiments/flow_transfer_direct_dense_obstacle3k_bc_skynet_normfix_20260827}
    ARMS=(
      bc_usocket_latent
      bc_usocket_dp
      bc_chain_latent
      bc_chain_dp
    )
    RESOURCE_SPECS=('hoffman-lab hoffman-lab a40 1 96G')
    ;;
  *)
    printf 'Unknown MATRIX_SCOPE=%s; use all or bc\n' "$MATRIX_SCOPE" >&2
    exit 64
    ;;
esac
STATE_DIR=$EXP_ROOT/submission
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH

EXPECTED_HEAD=$(git -C "$REPO" rev-parse HEAD)
EXPECTED_LAUNCHER_SHA=$(sha256sum "$LAUNCHER" | awk '{print $1}')
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$LATENT_NORM_ARTIFACT" | awk '{print $1}')" = "$LATENT_NORM_SHA"
test "$(sha256sum "$DP_NORM_ARTIFACT" | awk '{print $1}')" = "$DP_NORM_SHA"

mkdir -p "$STATE_DIR" "$EXP_ROOT/slurm" "$EXP_ROOT/runs" \
  "$EXP_ROOT/smokes" "$EXP_ROOT/provenance"
LOCK_DIR=$STATE_DIR/.submit_${PHASE}_lock
if ! mkdir "$LOCK_DIR"; then
  printf 'Submission lock already exists: %s\n' "$LOCK_DIR" >&2
  exit 73
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

case "$PHASE" in
  smoke)
    OUTPUT_TSV=$STATE_DIR/smoke_jobs.tsv
    MARKER=$STATE_DIR/SMOKES_SUBMITTED
    QOS=short
    TIME_LIMIT=02:00:00
    ;;
  full)
    OUTPUT_TSV=$STATE_DIR/full_jobs.tsv
    MARKER=$STATE_DIR/FULLS_SUBMITTED
    QOS=long
    TIME_LIMIT=7-00:00:00
    ;;
  *)
    printf 'Unknown PHASE=%s\n' "$PHASE" >&2
    exit 64
    ;;
esac
test ! -e "$MARKER"
test ! -e "$OUTPUT_TSV"
SMOKE_IDENTITY=$STATE_DIR/smoke_identity.tsv

if test "$PHASE" = smoke; then
  test ! -e "$SMOKE_IDENTITY"
  printf 'head\t%s\nlauncher_sha256\t%s\nlatent_norm_artifact\t%s\nlatent_norm_sha256\t%s\ndp_norm_artifact\t%s\ndp_norm_sha256\t%s\nenvironment_sha256\t%s\nmatrix_scope\t%s\n' \
    "$EXPECTED_HEAD" "$EXPECTED_LAUNCHER_SHA" \
    "$LATENT_NORM_ARTIFACT" "$LATENT_NORM_SHA" \
    "$DP_NORM_ARTIFACT" "$DP_NORM_SHA" "$EXPECTED_ENV_SHA" "$MATRIX_SCOPE" \
    > "$SMOKE_IDENTITY"
else
  test -s "$SMOKE_IDENTITY"
  identity_value() {
    awk -F '\t' -v key="$1" '$1 == key {print $2}' "$SMOKE_IDENTITY"
  }
  test "$(identity_value head)" = "$EXPECTED_HEAD"
  test "$(identity_value launcher_sha256)" = "$EXPECTED_LAUNCHER_SHA"
  test "$(identity_value latent_norm_artifact)" = "$LATENT_NORM_ARTIFACT"
  test "$(identity_value latent_norm_sha256)" = "$LATENT_NORM_SHA"
  test "$(identity_value dp_norm_artifact)" = "$DP_NORM_ARTIFACT"
  test "$(identity_value dp_norm_sha256)" = "$DP_NORM_SHA"
  test "$(identity_value environment_sha256)" = "$EXPECTED_ENV_SHA"
  test "$(identity_value matrix_scope)" = "$MATRIX_SCOPE"
fi

for resource_spec in "${RESOURCE_SPECS[@]}"; do
  read -r test_partition test_account test_gpu_type test_gpus test_memory \
    <<< "$resource_spec"
  "$SLURM_BIN/sbatch" --test-only \
    --partition="$test_partition" --account="$test_account" --qos="$QOS" \
    --nodes=1 --ntasks-per-node="$test_gpus" --cpus-per-task=8 \
    --mem="$test_memory" --time="$TIME_LIMIT" \
    --gres="gpu:$test_gpu_type:$test_gpus" --exclude=bishop --wrap=true
done

if test "$PHASE" = full; then
  test -s "$STATE_DIR/smoke_jobs.tsv"
  python - "$STATE_DIR/smoke_jobs.tsv" "$EXP_ROOT" "$MATRIX_SCOPE" <<'PY'
import csv
import json
import pathlib
import subprocess
import sys

jobs_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
matrix_scope = sys.argv[3]
rows = list(csv.DictReader(jobs_path.open(), delimiter="\t"))
expected = {
    "bc_usocket_latent": (1, "hoffman-lab", "hoffman-lab", "a40", 64),
    "bc_usocket_dp": (1, "hoffman-lab", "hoffman-lab", "a40", 64),
    "bc_chain_latent": (1, "hoffman-lab", "hoffman-lab", "a40", 64),
    "bc_chain_dp": (1, "hoffman-lab", "hoffman-lab", "a40", 64),
    "cotrain_obstacle_latent": (2, "rl2-lab", "rl2-lab", "l40s", 128),
    "cotrain_obstacle_dp": (2, "rl2-lab", "rl2-lab", "l40s", 128),
}
if matrix_scope == "bc":
    expected = {arm: value for arm, value in expected.items() if arm.startswith("bc_")}
else:
    assert matrix_scope == "all"
assert {row["arm"] for row in rows} == set(expected)
for row in rows:
    arm = row["arm"]
    job = row["job_id"]
    world_size, partition, account, gpu_type, total_global_batch = expected[arm]
    assert int(row["gpus"]) == world_size
    assert row["partition"] == partition
    assert row["account"] == account
    assert row["gpu_type"] == gpu_type
    accounting = subprocess.check_output(
        [
            "/opt/slurm/Ubuntu-20.04/24.11.0/bin/sacct",
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
    assert validation["expected_world_size"] == world_size
    assert validation["global_step"] == 3200
    assert validation["gradient_clipping_enabled"] is False
    assert set(validation["global_batch_per_domain"].values()) == {64}
    assert validation["total_global_batch"] == total_global_batch
PY
fi

: > "$OUTPUT_TSV"
printf 'arm\tjob_id\tpartition\taccount\tgpu_type\tgpus\tnorm_artifact\tnorm_sha256\n' >> "$OUTPUT_TSV"

for arm in "${ARMS[@]}"; do
  gpus=1
  memory=96G
  partition=hoffman-lab
  account=hoffman-lab
  gpu_type=a40
  if [[ "$arm" == cotrain_* ]]; then
    gpus=2
    memory=128G
    partition=rl2-lab
    account=rl2-lab
    gpu_type=l40s
  fi
  if [[ "$arm" == *_latent ]]; then
    norm_artifact=$LATENT_NORM_ARTIFACT
    norm_sha=$LATENT_NORM_SHA
  else
    norm_artifact=$DP_NORM_ARTIFACT
    norm_sha=$DP_NORM_SHA
  fi

  if test "$PHASE" = smoke; then
    job_prefix=ftsm
  else
    job_prefix=ft
  fi
  job_id=$(
    "$SLURM_BIN/sbatch" --parsable \
      --partition="$partition" --account="$account" --qos="$QOS" \
      --nodes=1 --ntasks-per-node="$gpus" --cpus-per-task=8 \
      --mem="$memory" --time="$TIME_LIMIT" \
      --gres="gpu:$gpu_type:$gpus" --exclude=bishop --requeue --signal=USR1@300 \
      --open-mode=append \
      --job-name="${job_prefix}_${arm:0:16}" \
      --output="$EXP_ROOT/slurm/${PHASE}_${arm}_%j.out" \
      --error="$EXP_ROOT/slurm/${PHASE}_${arm}_%j.err" \
      --export="ALL,ARM=$arm,MODE=$PHASE,GPUS_EXPECTED=$gpus,EXPECTED_HEAD=$EXPECTED_HEAD,EXPECTED_LAUNCHER_SHA=$EXPECTED_LAUNCHER_SHA,NORM_ARTIFACT=$norm_artifact,EXPECTED_NORM_SHA=$norm_sha,CLUSTER_PROFILE=skynet_world2,FLOW_TRANSFER_REPO=$REPO,FLOW_TRANSFER_EXP_ROOT=$EXP_ROOT,EXPECTED_ENV_SHA=$EXPECTED_ENV_SHA" \
      "$LAUNCHER"
  )
  job_id=${job_id%%;*}
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$job_id" "$partition" "$account" "$gpu_type" "$gpus" \
    "$norm_artifact" "$norm_sha" | tee -a "$OUTPUT_TSV"
done

date --iso-8601=seconds > "$MARKER"
printf '%s %s matrix submitted; full jobs are never chained automatically.\n' \
  "$PHASE" "$MATRIX_SCOPE"
