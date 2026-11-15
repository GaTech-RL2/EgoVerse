#!/usr/bin/env bash
set -Eeuo pipefail
set +x
[[ "${LEACT_WORKFLOW_ID:?}" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "${LEACT_PAYLOAD_SHA256:?}" =~ ^[a-f0-9]{64}$ ]]
[[ "${LEACT_ROLE:?}" =~ ^(self|frontier|astra)$ ]]
LEACT_REMOTE_BASE=/mnt/amlfs-07/shared/egoverse/rpunamiya/leact
LEACT_RUN_ROOT="$LEACT_REMOTE_BASE/runs/$LEACT_WORKFLOW_ID"
LEACT_SOURCE="$LEACT_RUN_ROOT/source"
mountpoint -q /mnt/amlfs-03/home/rpunamiya
LEACT_STUDY_ROOT="/mnt/amlfs-03/home/rpunamiya/leact/studies/overnight-20261022-$LEACT_ROLE"
LEACT_CHECKPOINT_STAGING="$LEACT_STUDY_ROOT/checkpoint-staging"
mkdir -p "$LEACT_RUN_ROOT" "$LEACT_STUDY_ROOT" "$LEACT_CHECKPOINT_STAGING"
exec > >(tee -a "$LEACT_RUN_ROOT/bootstrap.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%s,\"finished_utc\":\"%s\"}\n" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LEACT_RUN_ROOT/status.json"' EXIT
nvidia-smi --query-gpu=index,name,memory.total --format=csv
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
export TMPDIR="/tmp/leact-coldstart-$LEACT_ROLE"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
mkdir -p "$TMPDIR"
python -m pytest reproductions/leact/tests -q
python reproductions/leact/cluster/preflight.py --gpus 8 --output "$LEACT_STUDY_ROOT/runtime.json"
timeout --signal=TERM --kill-after=30 180 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 reproductions/leact/cluster/check_collectives.py
if [[ "$LEACT_ROLE" == astra ]]; then
    python -m reproductions.leact.astra_expert.run \
        --root "$LEACT_STUDY_ROOT" --teacher "$LEACT_SOURCE/study-input/teacher.jsonl" \
        --oracle "$LEACT_SOURCE/reproductions/leact/artifacts/leduc6r2s/oracle.jsonl" \
        --quality-gate "$LEACT_SOURCE/study-input/gate.json" \
        --anchor /mnt/amlfs-03/home/rpunamiya/leact/studies/overnight-20261022-frontier/anchor-ready.json \
        --scratch "$LEACT_CHECKPOINT_STAGING" --gpus 8 --rounds 1
else
    python -m reproductions.leact.coldstart.run \
        --role "$LEACT_ROLE" --root "$LEACT_STUDY_ROOT" \
        --oracle "$LEACT_SOURCE/reproductions/leact/artifacts/leduc6r2s/oracle.jsonl" \
        --saved-experiment "$LEACT_REMOTE_BASE/runs/leact-leduc-qwen3-8b-2/experiment-masked-bc-v1" \
        --frontier-input "$LEACT_SOURCE/study-input" \
        --scratch "$LEACT_CHECKPOINT_STAGING" --rounds 2
fi
for attempt in $(seq 1 36); do
    [[ -f "$LEACT_STUDY_ROOT/collection-exported.json" ]] && break
    sleep 5
done
