#!/bin/bash
# Control-plane wrapper around the immutable Flow Transfer launcher. It samples
# GPU memory without changing the training source or resolved Hydra config.

set -Eeuo pipefail

ARM=${ARM:?submit with ARM}
MODE=${MODE:?submit with MODE=smoke or MODE=full}
TRAIN_REPO=${TRAIN_REPO:?submit with immutable TRAIN_REPO}
TRAIN_LAUNCHER=${TRAIN_LAUNCHER:?submit with immutable TRAIN_LAUNCHER}
CONTROL_REPO=${CONTROL_REPO:?submit with CONTROL_REPO}
EXPECTED_HEAD=${EXPECTED_HEAD:?submit with immutable training HEAD}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?submit with training launcher SHA256}
EXPECTED_CONTROL_HEAD=${EXPECTED_CONTROL_HEAD:?submit with control HEAD}
EXPECTED_WRAPPER_SHA=${EXPECTED_WRAPPER_SHA:?submit with wrapper SHA256}
FLOW_TRANSFER_EXP_ROOT=${FLOW_TRANSFER_EXP_ROOT:?submit with experiment root}
FLOW_TRANSFER_PY_ENV=${FLOW_TRANSFER_PY_ENV:?submit with Python environment}
PACE_GPU_MODEL=${PACE_GPU_MODEL:?submit with A100 or H200}
A100_TOTAL_MIB=${A100_TOTAL_MIB:?submit with A100 capacity}
A100_SAFE_PEAK_MIB=${A100_SAFE_PEAK_MIB:?submit with safe peak threshold}

WRAPPER=$CONTROL_REPO/scripts/train/flow_transfer_chain_bc_memory_audit.sbatch
PROVENANCE_DIR=$FLOW_TRANSFER_EXP_ROOT/provenance/$MODE/$ARM/job_${SLURM_JOB_ID:?}
SAMPLES=$PROVENANCE_DIR/gpu_memory_samples_mib.tsv
RESULT=$PROVENANCE_DIR/memory_validation.json

test "$(git -C "$TRAIN_REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$TRAIN_REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$TRAIN_LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
test "$(git -C "$CONTROL_REPO" rev-parse HEAD)" = "$EXPECTED_CONTROL_HEAD"
test -z "$(git -C "$CONTROL_REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$WRAPPER" | awk '{print $1}')" = "$EXPECTED_WRAPPER_SHA"
test -x "$FLOW_TRANSFER_PY_ENV/bin/python"
test "$A100_TOTAL_MIB" = 81920
test "$A100_SAFE_PEAK_MIB" = 71680
test "$A100_SAFE_PEAK_MIB" -lt "$A100_TOTAL_MIB"

mkdir -p "$PROVENANCE_DIR"
test ! -e "$SAMPLES"
test ! -e "$RESULT"
: > "$SAMPLES"

"$TRAIN_LAUNCHER" &
train_pid=$!
monitor_pid=
cleanup_monitor() {
  if test -n "$monitor_pid" && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
  fi
  if test -n "$monitor_pid"; then
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap cleanup_monitor EXIT

(
  while kill -0 "$train_pid" 2>/dev/null; do
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR == 1 {print $1}')
    test -n "$used_mib"
    printf '%s\t%s\n' "$(date +%s)" "$used_mib" >> "$SAMPLES"
    sleep 1
  done
) &
monitor_pid=$!

set +e
wait "$train_pid"
train_status=$?
set -e
cleanup_monitor
monitor_pid=
trap - EXIT
test "$train_status" = 0

"$FLOW_TRANSFER_PY_ENV/bin/python" - \
  "$SAMPLES" "$RESULT" "$PROVENANCE_DIR" "$MODE" "$ARM" \
  "$PACE_GPU_MODEL" "$A100_TOTAL_MIB" "$A100_SAFE_PEAK_MIB" \
  "$EXPECTED_HEAD" "$EXPECTED_LAUNCHER_SHA" <<'PY'
import json
import pathlib
import sys

from omegaconf import OmegaConf

(
    samples_path,
    result_path,
    provenance_path,
    mode,
    arm,
    gpu_model,
    a100_total_mib,
    safe_peak_mib,
    source_head,
    launcher_sha,
) = sys.argv[1:]
samples_path = pathlib.Path(samples_path)
result_path = pathlib.Path(result_path)
provenance = pathlib.Path(provenance_path)
a100_total_mib = int(a100_total_mib)
safe_peak_mib = int(safe_peak_mib)

samples = []
for line in samples_path.read_text().splitlines():
    timestamp, used = line.split("\t")
    samples.append((int(timestamp), int(used.strip())))
assert samples
peak_mib = max(used for _, used in samples)
assert 0 < peak_mib <= safe_peak_mib, peak_mib
assert a100_total_mib - peak_mib >= 10_240, (a100_total_mib, peak_mib)

gpu_text = (provenance / "gpu.txt").read_text()
assert gpu_model.lower() in gpu_text.lower(), (gpu_model, gpu_text)
resolved = OmegaConf.load(provenance / "resolved_config.yaml")
assert resolved.trainer.precision == "bf16"
assert resolved.trainer.devices == 1
assert resolved.trainer.num_nodes == 1
assert resolved.trainer.accumulate_grad_batches == 1
assert resolved.trainer.get("gradient_clip_val") is None
assert set(resolved.data.train_datasets) == {"pushshapes_sim_chain_gripper"}
assert resolved.data.train_dataloader_params.pushshapes_sim_chain_gripper.batch_size == 64
assert resolved.trainer.limit_val_batches == (1 if mode == "smoke" else 0)
run_validation = json.loads((provenance / "run_validation.json").read_text())
assert run_validation["status"] == "PASS"
assert run_validation["mode"] == mode
assert run_validation["expected_world_size"] == 1
assert run_validation["total_global_batch"] == 64
assert run_validation["gradient_clipping_enabled"] is False

result = {
    "status": "PASS",
    "mode": mode,
    "arm": arm,
    "source_gpu_model": gpu_model,
    "source_head": source_head,
    "launcher_sha256": launcher_sha,
    "sample_count": len(samples),
    "peak_used_mib": peak_mib,
    "a100_total_mib": a100_total_mib,
    "a100_safe_peak_mib": safe_peak_mib,
    "a100_headroom_mib": a100_total_mib - peak_mib,
    "a100_memory_safe": True,
    "precision": "bf16",
    "world_size": 1,
    "global_batch": 64,
    "gradient_clipping_enabled": False,
}
result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY
