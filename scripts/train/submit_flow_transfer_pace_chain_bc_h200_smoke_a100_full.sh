#!/bin/bash
# Race the delayed A100 smokes on H200, then promote the exact same immutable
# training source to bounded A100 full jobs only after semantic and memory PASS.

set -Eeuo pipefail

PHASE=${PHASE:?set PHASE=smoke or PHASE=full}
EXPECTED_ENV_SHA=${EXPECTED_ENV_SHA:?set the exact PACE environment manifest SHA256}
LATENT_NORM_ARTIFACT=${LATENT_NORM_ARTIFACT:?exact horizon-100 norm_stats.json path}
LATENT_NORM_SHA=${LATENT_NORM_SHA:?exact horizon-100 norm SHA256}
DP_NORM_ARTIFACT=${DP_NORM_ARTIFACT:?exact horizon-16 norm_stats.json path}
DP_NORM_SHA=${DP_NORM_SHA:?exact horizon-16 norm SHA256}

TRAIN_REPO=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-flow-transfer-chain-bc-pace-20260827
CONTROL_REPO=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-flow-transfer-chain-bc-h200-race-20260827
PY_ENV=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse/.venv
DATA_ROOT=/storage/project/r-dxu345-0/paphiwetsa3/datasets/flow_transfer_20260826
U_DATA=$DATA_ROOT/u_socket_3000_v2_clean
CHAIN_OBSTACLE_ROOT=$DATA_ROOT/chain_gripper_obstacle_3000_balanced_v1
EXP_ROOT=/storage/project/r-dxu345-0/paphiwetsa3/experiments/flow_transfer_chain_bc_obstacle3k_pace_h200_smoke_a100_full_normfix_20260827
TRAIN_LAUNCHER=$TRAIN_REPO/scripts/train/flow_transfer_direct_dense_matrix.sbatch
HELPER=$CONTROL_REPO/scripts/train/submit_flow_transfer_pace_chain_bc_h200_smoke_a100_full.sh
WRAPPER=$CONTROL_REPO/scripts/train/flow_transfer_chain_bc_memory_audit.sbatch
STATE_DIR=$EXP_ROOT/submission
SLURM_BIN=/opt/slurm/current/bin
ACCOUNT=gts-dxu345-rl2
QOS=inferno
EXPECTED_TRAIN_HEAD=82bd2923a4daffb97f8f61713196422d0a811f00
EXPECTED_TRAIN_LAUNCHER_SHA=7ad60269d5ff08ba1bb4d290c913d1ae8f49b69b73f12696723bda48f6e2a1d0
SMOKE_GPU_TYPE=h200
SMOKE_GPU_MODEL=H200
SMOKE_PARTITION=gpu-h200
SMOKE_HOURS=2
H200_CREDITS_PER_GPU_HOUR=0.673
FULL_GPU_TYPE=a100
FULL_GPU_MODEL=A100
FULL_PARTITION=gpu-a100
FULL_GPU_CONSTRAINT=A100-80GB
FULL_HOURS=30
A100_CREDITS_PER_GPU_HOUR=0.276884
A100_TOTAL_MIB=81920
A100_SAFE_PEAK_MIB=71680
BC_MAX_TOTAL_CREDITS=20.0
export PATH=$SLURM_BIN:$PATH

case "$PHASE" in
  smoke)
    GPU_TYPE=$SMOKE_GPU_TYPE
    GPU_MODEL=$SMOKE_GPU_MODEL
    PARTITION=$SMOKE_PARTITION
    GPU_CONSTRAINT=
    TIME_LIMIT=02:00:00
    OUTPUT_TSV=$STATE_DIR/smoke_jobs.tsv
    MARKER=$STATE_DIR/SMOKES_SUBMITTED
    ;;
  full)
    GPU_TYPE=$FULL_GPU_TYPE
    GPU_MODEL=$FULL_GPU_MODEL
    PARTITION=$FULL_PARTITION
    GPU_CONSTRAINT=$FULL_GPU_CONSTRAINT
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

EXPECTED_CONTROL_HEAD=$(git -C "$CONTROL_REPO" rev-parse HEAD)
EXPECTED_HELPER_SHA=$(sha256sum "$HELPER" | awk '{print $1}')
EXPECTED_WRAPPER_SHA=$(sha256sum "$WRAPPER" | awk '{print $1}')
test "$(git -C "$TRAIN_REPO" rev-parse HEAD)" = "$EXPECTED_TRAIN_HEAD"
test -z "$(git -C "$TRAIN_REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$TRAIN_LAUNCHER" | awk '{print $1}')" = "$EXPECTED_TRAIN_LAUNCHER_SHA"
test -z "$(git -C "$CONTROL_REPO" status --porcelain=v1 --untracked-files=all)"
test -x "$PY_ENV/bin/python"
test -x "$SLURM_BIN/sbatch"
test -x "$SLURM_BIN/sacct"
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
  "$PY_ENV/bin/python" - "$SMOKE_HOURS" "$H200_CREDITS_PER_GPU_HOUR" \
    "$FULL_HOURS" "$A100_CREDITS_PER_GPU_HOUR" \
    "$BC_MAX_TOTAL_CREDITS" <<'PY'
import sys

smoke_hours, h200_rate, full_hours, a100_rate, cap = map(float, sys.argv[1:])
cost = 2 * smoke_hours * h200_rate + 2 * full_hours * a100_rate
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
  printf 'training_head\t%s\ntraining_launcher_sha256\t%s\ncontrol_head\t%s\nhelper_sha256\t%s\nwrapper_sha256\t%s\nlatent_norm_artifact\t%s\nlatent_norm_sha256\t%s\ndp_norm_artifact\t%s\ndp_norm_sha256\t%s\nenvironment_sha256\t%s\nsmoke_gpu_type\t%s\nsmoke_partition\t%s\nsmoke_hours_per_arm\t%s\nh200_credit_rate\t%s\nfull_gpu_type\t%s\nfull_partition\t%s\nfull_gpu_constraint\t%s\nfull_hours_per_arm\t%s\na100_credit_rate\t%s\na100_total_mib\t%s\na100_safe_peak_mib\t%s\nmaximum_total_credits\t%s\ncredit_cap\t%s\naccount\t%s\n' \
    "$EXPECTED_TRAIN_HEAD" "$EXPECTED_TRAIN_LAUNCHER_SHA" \
    "$EXPECTED_CONTROL_HEAD" "$EXPECTED_HELPER_SHA" "$EXPECTED_WRAPPER_SHA" \
    "$LATENT_NORM_ARTIFACT" "$LATENT_NORM_SHA" \
    "$DP_NORM_ARTIFACT" "$DP_NORM_SHA" "$EXPECTED_ENV_SHA" \
    "$SMOKE_GPU_TYPE" "$SMOKE_PARTITION" "$SMOKE_HOURS" \
    "$H200_CREDITS_PER_GPU_HOUR" "$FULL_GPU_TYPE" "$FULL_PARTITION" \
    "$FULL_GPU_CONSTRAINT" "$FULL_HOURS" "$A100_CREDITS_PER_GPU_HOUR" \
    "$A100_TOTAL_MIB" "$A100_SAFE_PEAK_MIB" "$EXPECTED_MAX_CREDITS" \
    "$BC_MAX_TOTAL_CREDITS" "$ACCOUNT" > "$SMOKE_IDENTITY"
else
  test -s "$SMOKE_IDENTITY"
  identity_value() {
    awk -F '\t' -v key="$1" '$1 == key {print $2}' "$SMOKE_IDENTITY"
  }
  test "$(identity_value training_head)" = "$EXPECTED_TRAIN_HEAD"
  test "$(identity_value training_launcher_sha256)" = "$EXPECTED_TRAIN_LAUNCHER_SHA"
  test "$(identity_value control_head)" = "$EXPECTED_CONTROL_HEAD"
  test "$(identity_value helper_sha256)" = "$EXPECTED_HELPER_SHA"
  test "$(identity_value wrapper_sha256)" = "$EXPECTED_WRAPPER_SHA"
  test "$(identity_value latent_norm_artifact)" = "$LATENT_NORM_ARTIFACT"
  test "$(identity_value latent_norm_sha256)" = "$LATENT_NORM_SHA"
  test "$(identity_value dp_norm_artifact)" = "$DP_NORM_ARTIFACT"
  test "$(identity_value dp_norm_sha256)" = "$DP_NORM_SHA"
  test "$(identity_value environment_sha256)" = "$EXPECTED_ENV_SHA"
  test "$(identity_value smoke_gpu_type)" = "$SMOKE_GPU_TYPE"
  test "$(identity_value smoke_partition)" = "$SMOKE_PARTITION"
  test "$(identity_value smoke_hours_per_arm)" = "$SMOKE_HOURS"
  test "$(identity_value h200_credit_rate)" = "$H200_CREDITS_PER_GPU_HOUR"
  test "$(identity_value full_gpu_type)" = "$FULL_GPU_TYPE"
  test "$(identity_value full_partition)" = "$FULL_PARTITION"
  test "$(identity_value full_gpu_constraint)" = "$FULL_GPU_CONSTRAINT"
  test "$(identity_value full_hours_per_arm)" = "$FULL_HOURS"
  test "$(identity_value a100_credit_rate)" = "$A100_CREDITS_PER_GPU_HOUR"
  test "$(identity_value a100_total_mib)" = "$A100_TOTAL_MIB"
  test "$(identity_value a100_safe_peak_mib)" = "$A100_SAFE_PEAK_MIB"
  test "$(identity_value maximum_total_credits)" = "$EXPECTED_MAX_CREDITS"
  test "$(identity_value credit_cap)" = "$BC_MAX_TOTAL_CREDITS"
  test "$(identity_value account)" = "$ACCOUNT"
fi

GPU_CONSTRAINT_ARGS=()
if test -n "$GPU_CONSTRAINT"; then
  GPU_CONSTRAINT_ARGS+=(--constraint="$GPU_CONSTRAINT")
fi
"$SLURM_BIN/sbatch" --test-only \
  --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=96G \
  --time="$TIME_LIMIT" --gres="gpu:$GPU_TYPE:1" \
  "${GPU_CONSTRAINT_ARGS[@]}" --wrap=true

ARMS=(bc_chain_latent bc_chain_dp)
if test "$PHASE" = full; then
  test -s "$STATE_DIR/smoke_jobs.tsv"
  "$PY_ENV/bin/python" - "$STATE_DIR/smoke_jobs.tsv" "$EXP_ROOT" \
    "$SMOKE_PARTITION" "$ACCOUNT" "$SMOKE_GPU_TYPE" \
    "$A100_TOTAL_MIB" "$A100_SAFE_PEAK_MIB" \
    "$EXPECTED_TRAIN_HEAD" "$EXPECTED_TRAIN_LAUNCHER_SHA" <<'PY'
import csv
import json
import pathlib
import subprocess
import sys

jobs_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
partition, account, gpu_type = sys.argv[3:6]
a100_total_mib, safe_peak_mib = map(int, sys.argv[6:8])
source_head, launcher_sha = sys.argv[8:10]
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
    provenance = root / "provenance" / "smoke" / arm / f"job_{job}"
    validation = json.loads((provenance / "run_validation.json").read_text())
    assert validation["status"] == "PASS"
    assert validation["mode"] == "smoke"
    assert validation["expected_world_size"] == 1
    assert validation["global_step"] == 3200
    assert validation["gradient_clipping_enabled"] is False
    assert validation["total_global_batch"] == 64
    assert validation["validation_enabled"] is True
    memory = json.loads((provenance / "memory_validation.json").read_text())
    assert memory["status"] == "PASS"
    assert memory["mode"] == "smoke"
    assert memory["arm"] == arm
    assert memory["source_gpu_model"] == "H200"
    assert memory["source_head"] == source_head
    assert memory["launcher_sha256"] == launcher_sha
    assert memory["a100_total_mib"] == a100_total_mib
    assert memory["a100_safe_peak_mib"] == safe_peak_mib
    assert memory["peak_used_mib"] <= safe_peak_mib
    assert memory["a100_headroom_mib"] >= 10_240
    assert memory["a100_memory_safe"] is True
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
  job_id=$(
    "$SLURM_BIN/sbatch" --parsable \
      --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
      --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=96G \
      --time="$TIME_LIMIT" --gres="gpu:$GPU_TYPE:1" \
      "${GPU_CONSTRAINT_ARGS[@]}" --no-requeue --open-mode=append \
      --job-name="ft${PHASE:0:1}_bc_${arm:0:16}" \
      --output="$EXP_ROOT/slurm/${PHASE}_${arm}_%j.out" \
      --error="$EXP_ROOT/slurm/${PHASE}_${arm}_%j.err" \
      --export="ALL,ARM=$arm,MODE=$PHASE,GPUS_EXPECTED=1,EXPECTED_HEAD=$EXPECTED_TRAIN_HEAD,EXPECTED_LAUNCHER_SHA=$EXPECTED_TRAIN_LAUNCHER_SHA,NORM_ARTIFACT=$norm_artifact,EXPECTED_NORM_SHA=$norm_sha,CLUSTER_PROFILE=pace_world1,FLOW_TRANSFER_REPO=$TRAIN_REPO,FLOW_TRANSFER_PY_ENV=$PY_ENV,FLOW_TRANSFER_EXP_ROOT=$EXP_ROOT,FLOW_TRANSFER_U_DATA=$U_DATA,CHAIN_OBSTACLE_ROOT=$CHAIN_OBSTACLE_ROOT,EXPECTED_ENV_SHA=$EXPECTED_ENV_SHA,PACE_GPU_MODEL=$GPU_MODEL,PACE_PARTITION=$PARTITION,TRAIN_REPO=$TRAIN_REPO,TRAIN_LAUNCHER=$TRAIN_LAUNCHER,CONTROL_REPO=$CONTROL_REPO,EXPECTED_CONTROL_HEAD=$EXPECTED_CONTROL_HEAD,EXPECTED_WRAPPER_SHA=$EXPECTED_WRAPPER_SHA,A100_TOTAL_MIB=$A100_TOTAL_MIB,A100_SAFE_PEAK_MIB=$A100_SAFE_PEAK_MIB" \
      "$WRAPPER"
  )
  job_id=${job_id%%;*}
  printf '%s\t%s\t%s\t%s\t%s\t1\t%s\t%s\n' \
    "$arm" "$job_id" "$PARTITION" "$ACCOUNT" "$GPU_TYPE" \
    "$norm_artifact" "$norm_sha" | tee -a "$OUTPUT_TSV"
done

date --iso-8601=seconds > "$MARKER"
printf '%s ChainGripper BC H200-smoke/A100-full cohort submitted; maximum charge is %s credits, and full jobs are never chained automatically.\n' \
  "$PHASE" "$EXPECTED_MAX_CREDITS"
