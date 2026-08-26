#!/bin/bash
# Submit the corrected six-arm smoke matrix, or submit full jobs only after
# every smoke has a semantic PASS. Normalization paths and SHAs are mandatory.

set -Eeuo pipefail

PHASE=${PHASE:?set PHASE=smoke or PHASE=full}
LATENT_NORM_ARTIFACT=${LATENT_NORM_ARTIFACT:?exact H100 norm_stats.json path}
LATENT_NORM_SHA=${LATENT_NORM_SHA:?exact H100 norm SHA256}
DP_NORM_ARTIFACT=${DP_NORM_ARTIFACT:?exact H16 norm_stats.json path}
DP_NORM_SHA=${DP_NORM_SHA:?exact H16 norm SHA256}
EXPECTED_ENV_SHA=8fd1504c955756adf8167f7bd34fc1a09cb844d268898b3ac30693e99ac60e87

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-direct-dense-obstacle-dp-schedulefix-20260826
LAUNCHER=$REPO/scripts/train/flow_transfer_direct_dense_matrix.sbatch
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_direct_dense_obstacle3k_cotrain_world2_normfix_20260826
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
  printf 'head\t%s\nlauncher_sha256\t%s\nlatent_norm_artifact\t%s\nlatent_norm_sha256\t%s\ndp_norm_artifact\t%s\ndp_norm_sha256\t%s\nenvironment_sha256\t%s\n' \
    "$EXPECTED_HEAD" "$EXPECTED_LAUNCHER_SHA" \
    "$LATENT_NORM_ARTIFACT" "$LATENT_NORM_SHA" \
    "$DP_NORM_ARTIFACT" "$DP_NORM_SHA" "$EXPECTED_ENV_SHA" \
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
fi

for resource_spec in '1 96G' '2 128G'; do
  read -r test_gpus test_memory <<< "$resource_spec"
  "$SLURM_BIN/sbatch" --test-only \
    --partition=rl2-lab --account=rl2-lab --qos="$QOS" \
    --nodes=1 --ntasks-per-node="$test_gpus" --cpus-per-task=8 \
    --mem="$test_memory" --time="$TIME_LIMIT" \
    --gres="gpu:l40s:$test_gpus" --exclude=bishop --wrap=true
done

ARMS=(
  bc_usocket_latent
  bc_usocket_dp
  bc_chain_latent
  bc_chain_dp
  cotrain_obstacle_latent
  cotrain_obstacle_dp
)

if test "$PHASE" = full; then
  test -s "$STATE_DIR/smoke_jobs.tsv"
  python - "$STATE_DIR/smoke_jobs.tsv" "$EXP_ROOT" <<'PY'
import csv
import json
import pathlib
import subprocess
import sys

jobs_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
rows = list(csv.DictReader(jobs_path.open(), delimiter="\t"))
expected = {
    "bc_usocket_latent": 1,
    "bc_usocket_dp": 1,
    "bc_chain_latent": 1,
    "bc_chain_dp": 1,
    "cotrain_obstacle_latent": 2,
    "cotrain_obstacle_dp": 2,
}
assert {row["arm"] for row in rows} == set(expected)
for row in rows:
    arm = row["arm"]
    job = row["job_id"]
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
    assert validation["expected_world_size"] == expected[arm]
    assert validation["global_step"] == 3200
    assert validation["gradient_clipping_enabled"] is False
PY
fi

: > "$OUTPUT_TSV"
printf 'arm\tjob_id\tgpus\tnorm_artifact\tnorm_sha256\n' >> "$OUTPUT_TSV"

for arm in "${ARMS[@]}"; do
  gpus=1
  memory=96G
  if [[ "$arm" == cotrain_* ]]; then
    gpus=2
    memory=128G
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
      --partition=rl2-lab --account=rl2-lab --qos="$QOS" \
      --nodes=1 --ntasks-per-node="$gpus" --cpus-per-task=8 \
      --mem="$memory" --time="$TIME_LIMIT" \
      --gres="gpu:l40s:$gpus" --exclude=bishop --requeue --signal=USR1@300 \
      --open-mode=append \
      --job-name="${job_prefix}_${arm:0:16}" \
      --output="$EXP_ROOT/slurm/${PHASE}_${arm}_%j.out" \
      --error="$EXP_ROOT/slurm/${PHASE}_${arm}_%j.err" \
      --export="ALL,ARM=$arm,MODE=$PHASE,GPUS_EXPECTED=$gpus,EXPECTED_HEAD=$EXPECTED_HEAD,EXPECTED_LAUNCHER_SHA=$EXPECTED_LAUNCHER_SHA,NORM_ARTIFACT=$norm_artifact,EXPECTED_NORM_SHA=$norm_sha" \
      "$LAUNCHER"
  )
  job_id=${job_id%%;*}
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$job_id" "$gpus" "$norm_artifact" "$norm_sha" | tee -a "$OUTPUT_TSV"
done

date --iso-8601=seconds > "$MARKER"
printf '%s matrix submitted; full jobs are never chained automatically.\n' "$PHASE"
