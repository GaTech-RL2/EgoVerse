#!/usr/bin/env bash
set -Eeuo pipefail
set +x
umask 0022

: "${LEACT_WORKFLOW_ID:?OSMO workflow ID is required}"
: "${LEACT_PAYLOAD_SHA256:?Build and hash the payload before submission}"
: "${LEACT_GPUS:=8}"
: "${LEACT_PHASE:=preflight}"
: "${LEACT_MODE:=bc}"
: "${LEACT_BASE:=/mnt/amlfs-07/shared/egoverse/rpunamiya/leact}"
[[ "$LEACT_PAYLOAD_SHA256" =~ ^[a-f0-9]{64}$ ]]
[[ "$LEACT_WORKFLOW_ID" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "$LEACT_GPUS" =~ ^(1|2|4|8)$ ]]
[[ "$LEACT_PHASE" =~ ^(preflight|full)$ ]]
[[ "$LEACT_MODE" =~ ^(bc|reasoning)$ ]]
[[ "$LEACT_BASE" == /mnt/amlfs-07/shared/egoverse/rpunamiya/leact ]]
test -d /mnt/amlfs-07/shared

export LEACT_RUN_ROOT="$LEACT_BASE/runs/$LEACT_WORKFLOW_ID"
export LEACT_EXPERIMENT_ROOT="${LEACT_EXPERIMENT_ROOT:-$LEACT_RUN_ROOT/experiment}"
[[ "$LEACT_EXPERIMENT_ROOT" == "$LEACT_BASE"/runs/* ]]
[[ "$LEACT_EXPERIMENT_ROOT" != *'/../'* && "$LEACT_EXPERIMENT_ROOT" != */.. ]]
export LEACT_SOURCE="$LEACT_RUN_ROOT/source"
mkdir -p "$LEACT_RUN_ROOT" "$LEACT_BASE/cache" /osmo/run/workspace
exec > >(tee -a "$LEACT_RUN_ROOT/bootstrap.log") 2>&1
phase=waiting_for_payload
record_exit() {
    local code=$?
    trap - EXIT
    printf '{"workflow_id":"%s","phase":"%s","exit_code":%s,"time_utc":"%s"}\n' \
        "$LEACT_WORKFLOW_ID" "$phase" "$code" "$(date -u +%FT%TZ)" \
        > "$LEACT_RUN_ROOT/bootstrap-status.json"
    exit "$code"
}
trap record_exit EXIT
printf 'Workflow %s; artifacts %s\n' "$LEACT_WORKFLOW_ID" "$LEACT_RUN_ROOT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h "$LEACT_BASE"

# The immutable archive arrives through native OSMO rsync or the PTY uploader.
# Only execute it after a complete transfer matches the submission's SHA256.
payload=/osmo/run/workspace/payload.tar.gz
ready=0
for attempt in $(seq 1 180); do
    if [[ -f "$payload" ]] && \
       [[ "$(sha256sum "$payload" | cut -d' ' -f1)" == "$LEACT_PAYLOAD_SHA256" ]]; then
        ready=1
        break
    fi
    sleep 5
done
if [[ "$ready" != 1 ]]; then
    echo 'No complete matching payload arrived within 15 minutes.'
    exit 2
fi
test ! -e "$LEACT_SOURCE"
mkdir "$LEACT_SOURCE"
cp "$payload" "$LEACT_RUN_ROOT/payload.tar.gz"
printf '%s  payload.tar.gz\n' "$LEACT_PAYLOAD_SHA256" > "$LEACT_RUN_ROOT/payload.sha256"
tar -xzf "$payload" -C "$LEACT_SOURCE" --no-same-owner
cd "$LEACT_SOURCE"
test -f reproductions/leact/experiment.py
test -f reproductions/leact/artifacts/leduc6r2s/oracle.jsonl

phase=installing_environment
export UV_CACHE_DIR="$LEACT_BASE/cache/uv"
export HF_HOME="$LEACT_BASE/cache/huggingface"
export HF_DATASETS_CACHE="$LEACT_RUN_ROOT/datasets-cache"
export UV_INDEX_URL=https://pypi.org/simple
export UV_INDEX_STRATEGY=first-index
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=300
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset PIP_CONSTRAINT UV_CONSTRAINT UV_BUILD_CONSTRAINT PIP_BUILD_CONSTRAINT
export PYTHONPATH="$LEACT_SOURCE"
export TRITON_CACHE_DIR="/tmp/leact-$LEACT_WORKFLOW_ID/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/leact-$LEACT_WORKFLOW_ID/inductor"
export VLLM_CACHE_ROOT="/tmp/leact-$LEACT_WORKFLOW_ID/vllm"
# Independent TP=1 workers raced exporting into the same rank_0_0 cache.
# vLLM 0.10.2 checks this flag before the failing compiled_graph.save call.
export VLLM_DISABLE_COMPILE_CACHE=1
printf 'Runtime override: VLLM_DISABLE_COMPILE_CACHE=%s\n' "$VLLM_DISABLE_COMPILE_CACHE"
export CUDA_CACHE_PATH="/tmp/leact-$LEACT_WORKFLOW_ID/cuda"
export WANDB_MODE=disabled

# The digest-pinned image supplies uv and Python 3.12 for environment creation.
# All project Python commands below run after activating emimic.
uv --no-config venv --python /usr/bin/python3.12 emimic
source emimic/bin/activate
uv --no-config pip install --python "$VIRTUAL_ENV/bin/python" --require-hashes \
    -r reproductions/leact/cluster/bootstrap-uv.txt
python -m uv --no-config pip sync --python "$VIRTUAL_ENV/bin/python" --require-hashes --no-build \
    reproductions/leact/cluster/requirements-gpu.lock
python -m uv --no-config pip check --python "$VIRTUAL_ENV/bin/python"
python -m uv --no-config pip freeze --python "$VIRTUAL_ENV/bin/python" \
    > "$LEACT_RUN_ROOT/environment.freeze.txt"
cp reproductions/leact/cluster/requirements-gpu.lock "$LEACT_RUN_ROOT/requirements-gpu.lock"

phase=checking_runtime
python -m pytest reproductions/leact/tests -q
python reproductions/leact/cluster/preflight.py \
    --gpus "$LEACT_GPUS" --output "$LEACT_RUN_ROOT/runtime.json"
timeout --signal=TERM --kill-after=30 180 \
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$LEACT_GPUS" \
    reproductions/leact/cluster/check_collectives.py

phase=experiment
# Sequential reads avoid slow mmap page faults on a newly allocated node.
# This only warms the host file cache; the saved checkpoint is not modified.
if [[ "$LEACT_PHASE" == full && -f "$LEACT_EXPERIMENT_ROOT/coldstart/training_complete.json" ]]; then
    printf 'Warming completed coldstart checkpoint from shared storage\n'
    if ! timeout 120 bash -e -c '
        for leact_checkpoint_file in "$1"/*.safetensors; do
            dd if="$leact_checkpoint_file" of=/dev/null bs=16M status=none
        done
    ' _ "$LEACT_EXPERIMENT_ROOT/coldstart/checkpoints"; then
        printf 'Checkpoint pre-read incomplete; continuing with normal model loading\n'
    fi
fi
experiment_args=(--root "$LEACT_EXPERIMENT_ROOT" --gpus "$LEACT_GPUS" --phase "$LEACT_PHASE" --mode "$LEACT_MODE")
if [[ -n "${LEACT_BC_CHECKPOINT_ROOT:-}" ]]; then
    experiment_args+=(--bc-checkpoint-root "$LEACT_BC_CHECKPOINT_ROOT")
fi
if [[ "$LEACT_PHASE" == full && "$LEACT_MODE" == reasoning ]]; then
    : "${LEACT_REASONING_CHECKPOINT:?Explicit reasoning checkpoint is required}"
    : "${LEACT_REASONING_PROVENANCE:?Explicit corpus provenance is required}"
    experiment_args+=(--reasoning-checkpoint "$LEACT_REASONING_CHECKPOINT" --reasoning-provenance "$LEACT_REASONING_PROVENANCE")
fi
timeout --signal=TERM --kill-after=300 "${LEACT_EXPERIMENT_TIMEOUT:-10h}" \
    python -m reproductions.leact.experiment \
    "${experiment_args[@]}"
phase=complete
