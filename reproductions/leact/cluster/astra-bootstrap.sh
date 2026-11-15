#!/usr/bin/env bash
set -Eeuo pipefail
set +x
[[ "${LEACT_WORKFLOW_ID:?}" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "${LEACT_PAYLOAD_SHA256:?}" =~ ^[a-f0-9]{64}$ ]]
LEACT_REMOTE_BASE=/mnt/amlfs-07/shared/egoverse/rpunamiya/leact
LEACT_RUN_ROOT="$LEACT_REMOTE_BASE/runs/$LEACT_WORKFLOW_ID"
LEACT_SOURCE="$LEACT_RUN_ROOT/source"
mkdir -p "$LEACT_RUN_ROOT"
exec > >(tee -a "$LEACT_RUN_ROOT/bootstrap.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s,\"finished_utc\":\"%s\"}\n" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LEACT_RUN_ROOT/status.json"' EXIT
ready=0
for attempt in $(seq 1 180); do
    if [[ -f /osmo/run/workspace/payload.tar.gz ]] && [[ "$(sha256sum /osmo/run/workspace/payload.tar.gz | cut -d' ' -f1)" == "$LEACT_PAYLOAD_SHA256" ]]; then
        ready=1
        break
    fi
    sleep 5
done
[[ "$ready" == 1 ]]
test ! -e "$LEACT_SOURCE"
mkdir "$LEACT_SOURCE"
cp /osmo/run/workspace/payload.tar.gz "$LEACT_RUN_ROOT/payload.tar.gz"
tar -xzf /osmo/run/workspace/payload.tar.gz -C "$LEACT_SOURCE" --no-same-owner
ln -s "$LEACT_REMOTE_BASE/runs/leact-leduc-qwen3-8b-7/source/emimic" "$LEACT_SOURCE/emimic"
cd "$LEACT_SOURCE"
source emimic/bin/activate
export PYTHONPATH="$LEACT_SOURCE"
export TMPDIR=/tmp/leact-astra-v2
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
mkdir -p "$TMPDIR"
python -m pytest reproductions/leact/tests -q
python reproductions/leact/cluster/preflight.py --gpus 8 --output "$LEACT_RUN_ROOT/runtime.json"
timeout --signal=TERM --kill-after=30 180 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 reproductions/leact/cluster/check_collectives.py
python -m reproductions.leact.astra_expert.run \
    --root "$LEACT_REMOTE_BASE/studies/astra-conditioning-v2" \
    --teacher "$LEACT_SOURCE/queued-input/teacher.jsonl" \
    --oracle "$LEACT_SOURCE/reproductions/leact/artifacts/leduc6r2s/oracle.jsonl" \
    --leduc-results "$LEACT_SOURCE/queued-input/leduc-results.json" \
    --anchor "$LEACT_REMOTE_BASE/studies/concise-reproduction-v1/anchor/sft/checkpoints" \
    --scratch /osmo/run/workspace/astra-checkpoints --gpus 8 --rounds 3
# Leave time for the existing authenticated collector to acknowledge small results.
for attempt in $(seq 1 24); do
    [[ -f "$LEACT_REMOTE_BASE/studies/astra-conditioning-v2/collection-exported.json" ]] && break
    sleep 5
done
