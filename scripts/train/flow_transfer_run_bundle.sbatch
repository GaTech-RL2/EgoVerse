#!/bin/bash
# Generic immutable-bundle launcher for Flow Transfer norm, smoke, and full runs.
# Resource flags are deliberately supplied by sbatch from the pinned bundle.

set -Eeuo pipefail

MODE=${MODE:?set MODE=norm, smoke, or full}
REPO=${REPO:?set the exact clean repository checkout}
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
PY_ENV=${PY_ENV:?set the pinned Python environment}
SLURM_BIN=${SLURM_BIN:-/opt/slurm/Ubuntu-20.04/24.11.0/bin}
export PATH=$SLURM_BIN:$PATH

case "$MODE" in
  norm)
    MANIFEST=${RUN_SPEC:?set the staged run spec}
    ;;
  smoke|full)
    MANIFEST=${RUN_BUNDLE:?set the finalized bundle manifest}
    ;;
  *)
    printf 'Unsupported MODE=%s\n' "$MODE" >&2
    exit 2
    ;;
esac

PYTHON=$PY_ENV/bin/python
TOOL=$REPO/egomimic/scripts/flow_transfer_run_bundle.py
SMOKE_VERIFIER=$REPO/egomimic/scripts/verify_training_smoke.py
FULL_VERIFIER=$REPO/egomimic/scripts/verify_training_full.py
SRUN=$SLURM_BIN/srun

test -x "$PYTHON"
test -x "$SRUN"
test -f "$TOOL"
test -f "$SMOKE_VERIFIER"
test -f "$FULL_VERIFIER"

field() {
  "$PYTHON" "$TOOL" print-field --manifest "$MANIFEST" --field "$1"
}

EXPECTED_HEAD=$(field source.head)
EXPECTED_REPO=$(field source.repo)
EXPECTED_ENV=$(field python_environment.path)
OUTPUT_ROOT=$(field outputs.root)
WORLD_SIZE=$(field resources.world_size)
ACCOUNT=$(field resources.account)
PARTITION=$(field resources.partition)
GPU_TYPE=$(field resources.gpu_type)
CPUS_PER_TASK=$(field resources.cpus_per_task)

test "$(realpath "$REPO")" = "$(realpath "$EXPECTED_REPO")"
test "$(realpath "$PY_ENV")" = "$(realpath "$EXPECTED_ENV")"
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_JOB_PARTITION:?}" = "$PARTITION"
test "${SLURM_JOB_ACCOUNT:?}" = "$ACCOUNT"
test "${SLURM_CPUS_PER_TASK:?}" -ge "$CPUS_PER_TASK"

if test "$MODE" = norm; then
  test "${SLURM_NTASKS:?}" = 1
else
  test "${SLURM_NTASKS:?}" = "$WORLD_SIZE"
  test -n "${CUDA_VISIBLE_DEVICES:-}"
  IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  test "${#VISIBLE_GPUS[@]}" = "$WORLD_SIZE"
fi

"$PYTHON" "$TOOL" verify --manifest "$MANIFEST" --phase "$MODE"

if test "$MODE" = full; then
  PROVENANCE=$OUTPUT_ROOT/provenance/$MODE/job_${SLURM_JOB_ID:?}/attempt_${SLURM_RESTART_COUNT:-0}
else
  PROVENANCE=$OUTPUT_ROOT/provenance/$MODE/job_${SLURM_JOB_ID:?}
fi
test ! -e "$PROVENANCE"
mkdir -p "$PROVENANCE"
cp "$REPO/scripts/train/flow_transfer_run_bundle.sbatch" "$PROVENANCE/launcher.sbatch"
"$SLURM_BIN/scontrol" show job -dd "$SLURM_JOB_ID" > "$PROVENANCE/slurm_job.txt"
git -C "$REPO" show -s --format=fuller "$EXPECTED_HEAD" > "$PROVENANCE/git_commit.txt"
git -C "$REPO" status --porcelain=v1 --untracked-files=all \
  > "$PROVENANCE/git_status.txt"
test ! -s "$PROVENANCE/git_status.txt"

if test "$MODE" = full; then
  test "${FULL_SIGNAL:?set the exact sbatch --signal contract}" = \
    "$(field resources.full_signal)"
  printf '%s\n' "$FULL_SIGNAL" > "$PROVENANCE/full_signal.txt"
  EXPECTED_QOS=$(field resources.full_qos)
  EXPECTED_TIME=$(field resources.full_time)
  EXPECTED_REQUEUE=1
else
  EXPECTED_QOS=$(field resources.smoke_qos)
  EXPECTED_TIME=$(field resources.smoke_time)
  EXPECTED_REQUEUE=0
fi
"$PYTHON" - "$PROVENANCE/slurm_job.txt" "$EXPECTED_QOS" \
  "$EXPECTED_TIME" "$EXPECTED_REQUEUE" "$(field resources.memory)" <<'PY'
import pathlib
import re
import sys

path, qos, time_limit, requeue, memory = sys.argv[1:]
fields = dict(
    re.findall(r"([A-Za-z][A-Za-z0-9/]*)=([^\s]+)", pathlib.Path(path).read_text())
)
assert fields["QOS"] == qos, (fields.get("QOS"), qos)
assert fields["TimeLimit"] == time_limit, (fields.get("TimeLimit"), time_limit)
assert fields["Requeue"] == requeue, (fields.get("Requeue"), requeue)
assert f"mem={memory}" in fields["ReqTRES"].split(","), fields["ReqTRES"]
PY

if test "$MODE" != norm; then
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader > "$PROVENANCE/gpu.txt"
  test "$(wc -l < "$PROVENANCE/gpu.txt")" = "$WORLD_SIZE"
  test "$(grep -ci "$GPU_TYPE" "$PROVENANCE/gpu.txt")" = "$WORLD_SIZE"
fi

mapfile -d '' -t HYDRA_ARGS < <(
  "$PYTHON" "$TOOL" emit-args --manifest "$MANIFEST" --phase "$MODE"
)
test "${#HYDRA_ARGS[@]}" -gt 5

source "$PY_ENV/bin/activate"
cd "$REPO"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true

write_progress() {
  local status=$?
  if test "$MODE" = full; then
    "$PYTHON" - "$OUTPUT_ROOT" "$SLURM_JOB_ID" "$status" <<'PY' || true
import csv
import pathlib
import sys

import torch

root = pathlib.Path(sys.argv[1])
job_id = sys.argv[2]
exit_code = int(sys.argv[3])
run = root / "full" / "run"
candidates = []
for pattern in ("hpc_ckpt_*.ckpt", "checkpoints/*.ckpt", "checkpoints/final/*.ckpt"):
    candidates.extend(run.glob(pattern))
step = 0
for path in candidates:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        step = max(step, int(payload.get("global_step", 0)))
    except Exception:
        continue
destination = root / "monitor" / f"progress_{job_id}.csv"
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["step", "exit_code"])
    writer.writerow([step, exit_code])
PY
  fi
  return "$status"
}
trap write_progress EXIT

case "$MODE" in
  norm)
    RUN_DIR=$(field outputs.norm_run)
    test ! -e "$RUN_DIR"
    mkdir -p "$(dirname "$RUN_DIR")"
    export WANDB_MODE=offline
    "$PYTHON" -m egomimic.trainHydra "${HYDRA_ARGS[@]}" \
      2>&1 | tee "$OUTPUT_ROOT/norm/training.log"
    "$PYTHON" "$TOOL" finalize-norm --spec "$MANIFEST"
    ;;
  smoke)
    RUN_DIR=$(field outputs.smoke_run)
    test ! -e "$RUN_DIR"
    mkdir -p "$(dirname "$RUN_DIR")"
    export WANDB_MODE=offline
    "$SRUN" --kill-on-bad-exit=1 --unbuffered \
      "$PYTHON" -m egomimic.trainHydra "${HYDRA_ARGS[@]}" \
      2>&1 | tee "$OUTPUT_ROOT/smoke/training.log"
    EMBODIMENTS=$(
      "$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
domains = payload["normalization"]["contract"]["domains"].values()
print(",".join(str(item["embodiment_id"]) for item in domains))
PY
    )
    CUDA_VISIBLE_DEVICES= "$PYTHON" "$SMOKE_VERIFIER" "$RUN_DIR" \
      --required-embodiments "$EMBODIMENTS" \
      --expected-head "$EXPECTED_HEAD" \
      --expected-world-size "$WORLD_SIZE" \
      --dry-run
    "$PYTHON" "$TOOL" verify --manifest "$MANIFEST" --phase smoke
    CUDA_VISIBLE_DEVICES= "$PYTHON" "$SMOKE_VERIFIER" "$RUN_DIR" \
      --required-embodiments "$EMBODIMENTS" \
      --expected-head "$EXPECTED_HEAD" \
      --expected-world-size "$WORLD_SIZE"
    "$PYTHON" "$TOOL" verify-smoke-result \
      --manifest "$MANIFEST" \
      --result "$RUN_DIR/SMOKE_RESULT.json"
    ;;
  full)
    "$PYTHON" "$TOOL" verify-smoke-gate --manifest "$MANIFEST"
    RUN_DIR=$(field outputs.full_run)
    FRESH_RUN=false
    if test ! -e "$RUN_DIR"; then
      mkdir -p "$RUN_DIR"
      FRESH_RUN=true
    else
      test -d "$RUN_DIR"
    fi
    test ! -e "$RUN_DIR/FULL_RESULT.json"
    export WANDB_MODE=online

    RESUME_CKPT=$(
      "$PYTHON" - "$RUN_DIR" "$MANIFEST" <<'PY'
import json
import pathlib
import sys

import torch
from omegaconf import OmegaConf

run = pathlib.Path(sys.argv[1])
manifest = json.load(open(sys.argv[2]))
pinned = OmegaConf.load(manifest["configs"]["full"]["path"])
expected_model = OmegaConf.to_container(pinned.model, resolve=True)
expected_provenance = OmegaConf.to_container(pinned.run_provenance, resolve=True)
expected_max_step = int(manifest["contract"]["training"]["max_steps"])
candidates = []
for pattern in ("hpc_ckpt_*.ckpt", "checkpoints/*.ckpt", "checkpoints/final/*.ckpt"):
    candidates.extend(run.glob(pattern))
valid = []
for path in set(candidates):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload.get("state_dict")
        assert payload.get("optimizer_states")
        step = int(payload.get("global_step", 0))
        assert 0 < step <= expected_max_step
        config_tree = payload["hyper_parameters"]["config_tree"]
        assert OmegaConf.to_container(
            OmegaConf.create(config_tree["model"]), resolve=True
        ) == expected_model
        assert OmegaConf.to_container(
            OmegaConf.create(config_tree["run_provenance"]), resolve=True
        ) == expected_provenance
        valid.append((step, path.stat().st_mtime_ns, path))
    except Exception:
        continue
if valid:
    print(max(valid)[2])
PY
    )
    if test -n "$RESUME_CKPT"; then
      printf '[full] resuming from %s\n' "$RESUME_CKPT"
      test "$FRESH_RUN" = false
      printf '%s\n' "$RESUME_CKPT" > "$PROVENANCE/resume_checkpoint.txt"
      sha256sum "$RESUME_CKPT" > "$PROVENANCE/resume_checkpoint.sha256"
      RESUME_ARG_REPLACED=0
      for index in "${!HYDRA_ARGS[@]}"; do
        if test "${HYDRA_ARGS[$index]}" = "++logger.wandb.resume=never"; then
          HYDRA_ARGS[$index]="++logger.wandb.resume=allow"
          RESUME_ARG_REPLACED=$((RESUME_ARG_REPLACED + 1))
        fi
      done
      test "$RESUME_ARG_REPLACED" = 1
      HYDRA_ARGS+=("ckpt_path=$RESUME_CKPT")
    else
      test "$FRESH_RUN" = true
    fi

    "$SRUN" --kill-on-bad-exit=1 --unbuffered \
      "$PYTHON" -m egomimic.trainHydra "${HYDRA_ARGS[@]}" \
      2>&1 | tee -a "$OUTPUT_ROOT/full/training.log"

    "$PYTHON" "$TOOL" verify-smoke-gate --manifest "$MANIFEST"
    CONFIG=$RUN_DIR/.hydra/config.yaml
    NORM=$(field outputs.norm_artifact)
    SMOKE=$(field outputs.smoke_run)/SMOKE_RESULT.json
    MAX_STEP=$(field contract.training.max_steps)
    CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
    NORM_SHA=$(sha256sum "$NORM" | awk '{print $1}')
    BUNDLE_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
    SMOKE_SHA=$(sha256sum "$SMOKE" | awk '{print $1}')
    CUDA_VISIBLE_DEVICES= "$PYTHON" "$FULL_VERIFIER" "$RUN_DIR" \
      --expected-max-step "$MAX_STEP" \
      --expected-head "$EXPECTED_HEAD" \
      --expected-world-size "$WORLD_SIZE" \
      --resolved-config "$CONFIG" \
      --expected-config-sha256 "$CONFIG_SHA" \
      --norm-artifact "$NORM" \
      --expected-norm-sha256 "$NORM_SHA" \
      --bundle-manifest "$MANIFEST" \
      --expected-bundle-sha256 "$BUNDLE_SHA" \
      --smoke-result "$SMOKE" \
      --expected-smoke-result-sha256 "$SMOKE_SHA" \
      --dry-run
    "$PYTHON" "$TOOL" verify --manifest "$MANIFEST" --phase full
    CUDA_VISIBLE_DEVICES= "$PYTHON" "$FULL_VERIFIER" "$RUN_DIR" \
      --expected-max-step "$MAX_STEP" \
      --expected-head "$EXPECTED_HEAD" \
      --expected-world-size "$WORLD_SIZE" \
      --resolved-config "$CONFIG" \
      --expected-config-sha256 "$CONFIG_SHA" \
      --norm-artifact "$NORM" \
      --expected-norm-sha256 "$NORM_SHA" \
      --bundle-manifest "$MANIFEST" \
      --expected-bundle-sha256 "$BUNDLE_SHA" \
      --smoke-result "$SMOKE" \
      --expected-smoke-result-sha256 "$SMOKE_SHA"
    ;;
esac

printf '[%s] ALL_PASS head=%s root=%s\n' "$MODE" "$EXPECTED_HEAD" "$OUTPUT_ROOT"
