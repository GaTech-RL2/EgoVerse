#!/usr/bin/env bash
set -Eeuo pipefail
set +x
: "${LEACT_WORKFLOW_ID:?}"
: "${LEACT_PAYLOAD_SHA256:?}"
[[ "$LEACT_PAYLOAD_SHA256" =~ ^[a-f0-9]{64}$ ]]
[[ "$LEACT_WORKFLOW_ID" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "$LEACT_BASE" == /mnt/amlfs-07/shared/egoverse/rpunamiya/leact ]]
export LEACT_RUN_ROOT="$LEACT_BASE/runs/$LEACT_WORKFLOW_ID"
export LEACT_SOURCE="$LEACT_RUN_ROOT/source"
export LEACT_DIAGNOSTIC_ROOT="$LEACT_RUN_ROOT/prompt-diagnostic"
mkdir -p "$LEACT_RUN_ROOT" /osmo/run/workspace
exec > >(tee -a "$LEACT_RUN_ROOT/bootstrap.log") 2>&1
phase=waiting_for_payload
trap 'code=$?; printf "{\"exit_code\":%s,\"phase\":\"%s\",\"finished_utc\":\"%s\"}\n" "$code" "$phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LEACT_RUN_ROOT/status.json"' EXIT
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h "$LEACT_BASE"
payload=/osmo/run/workspace/payload.tar.gz
ready=0
for attempt in $(seq 1 180); do
    if [[ -f "$payload" ]] && [[ "$(sha256sum "$payload" | cut -d' ' -f1)" == "$LEACT_PAYLOAD_SHA256" ]]; then
        ready=1
        break
    fi
    sleep 5
done
[[ "$ready" == 1 ]]
test ! -e "$LEACT_SOURCE"
mkdir "$LEACT_SOURCE"
cp "$payload" "$LEACT_RUN_ROOT/payload.tar.gz"
printf '%s  payload.tar.gz\n' "$LEACT_PAYLOAD_SHA256" > "$LEACT_RUN_ROOT/payload.sha256"
tar -xzf "$payload" -C "$LEACT_SOURCE" --no-same-owner
test -f "$LEACT_ENV_SOURCE/emimic/bin/activate"
ln -s "$LEACT_ENV_SOURCE/emimic" "$LEACT_SOURCE/emimic"
cd "$LEACT_SOURCE"
source emimic/bin/activate
export PYTHONPATH="$LEACT_SOURCE"
export TMPDIR=/tmp/leact-prompt
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$TMPDIR" "$LEACT_DIAGNOSTIC_ROOT"
phase=checking_runtime
# Job2 passed all 34 tests in this same immutable environment. Reuse that
# evidence only after checking every packaged Python source/test byte-for-byte.
python - <<'PY'
import os
from pathlib import Path
current = Path('reproductions/leact')
verified = Path(os.environ['LEACT_BASE'])/'runs/leact-prompt-diagnostic-2/source/reproductions/leact'
files = sorted(current.rglob('*.py'))
for path in files:
    if path.read_bytes() != (verified/path.relative_to(current)).read_bytes():
        raise RuntimeError(f'Source differs from the snapshot that passed 34 tests: {path}')
print(f'Reused 34/34 GPU-environment tests from job2; verified {len(files)} Python source/test files unchanged.', flush=True)
PY
python reproductions/leact/cluster/preflight.py --gpus "$LEACT_GPUS" --output "$LEACT_DIAGNOSTIC_ROOT/runtime.json"
phase=prompt_experiment
timeout --signal=TERM --kill-after=30 30m python -m reproductions.leact.prompt_diagnostic \
    --root "$LEACT_DIAGNOSTIC_ROOT" \
    --oracle "$LEACT_SOURCE/reproductions/leact/artifacts/leduc6r2s/oracle.jsonl" \
    --saved-experiment "$LEACT_SAVED_EXPERIMENT" \
    --bc-checkpoint "$LEACT_SAVED_EXPERIMENT/coldstart/checkpoints" --gpus "$LEACT_GPUS"
phase=collecting_results
# Keep the OSMO exec endpoint available for the checksum collector, bounded to 2m.
for attempt in $(seq 1 24); do
    [[ -f "$LEACT_DIAGNOSTIC_ROOT/collection-exported.json" ]] && break
    sleep 5
done
sleep 5
phase=complete
