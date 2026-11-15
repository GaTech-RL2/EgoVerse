#!/usr/bin/env bash
set -Eeuo pipefail
set +x
: "${LEACT_WORKFLOW_ID:?}"
: "${LEACT_PAYLOAD_SHA256:?}"
: "${LEACT_ROLE:?}"
[[ "$LEACT_PAYLOAD_SHA256" =~ ^[a-f0-9]{64}$ ]]
[[ "$LEACT_WORKFLOW_ID" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "$LEACT_ROLE" =~ ^(leact|exit)$ ]]
[[ "$LEACT_GPUS" == 8 ]]
[[ "$LEACT_BASE" == /mnt/amlfs-07/shared/egoverse/rpunamiya/leact ]]
[[ "$LEACT_STUDY_ROOT" == "$LEACT_BASE/studies/concise-reproduction-v1" ]]
export LEACT_RUN_ROOT="$LEACT_BASE/runs/$LEACT_WORKFLOW_ID"
export LEACT_SOURCE="$LEACT_RUN_ROOT/source"
export LEACT_ROLE_ROOT="$LEACT_STUDY_ROOT/$LEACT_ROLE"
mkdir -p "$LEACT_RUN_ROOT" /osmo/run/workspace
exec > >(tee -a "$LEACT_RUN_ROOT/bootstrap.log") 2>&1
phase=waiting_for_payload
trap 'code=$?; printf "{\"exit_code\":%s,\"phase\":\"%s\",\"role\":\"%s\",\"finished_utc\":\"%s\"}\n" "$code" "$phase" "$LEACT_ROLE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LEACT_RUN_ROOT/status.json"' EXIT
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
phase=preparing_source
test ! -e "$LEACT_SOURCE"
mkdir "$LEACT_SOURCE"
cp "$payload" "$LEACT_RUN_ROOT/payload.tar.gz"
printf '%s  payload.tar.gz\n' "$LEACT_PAYLOAD_SHA256" > "$LEACT_RUN_ROOT/payload.sha256"
tar -xzf "$payload" -C "$LEACT_SOURCE" --no-same-owner
test -f "$LEACT_ENV_SOURCE/emimic/bin/activate"
test -d "$LEACT_SAVED_EXPERIMENT"
test -d "$LEACT_PROMPT_ROOT"
ln -s "$LEACT_ENV_SOURCE/emimic" "$LEACT_SOURCE/emimic"
cd "$LEACT_SOURCE"
source emimic/bin/activate
export PYTHONPATH="$LEACT_SOURCE"
export TMPDIR="/tmp/leact-reproduction-$LEACT_ROLE"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export XDG_CACHE_HOME="$TMPDIR/xdg"
mkdir -p "$TMPDIR" "$LEACT_ROLE_ROOT"
mountpoint -q /mnt/amlfs-03/home/rpunamiya
LEACT_CHECKPOINT_STAGING="/mnt/amlfs-03/home/rpunamiya/leact/studies/concise-reproduction-v1/checkpoint-staging-$LEACT_ROLE"
mkdir -p "$LEACT_CHECKPOINT_STAGING"
# Preserve resolution of checkpoint symlinks created by earlier job attempts.
LEACT_LEGACY_SCRATCH="/osmo/run/workspace/leact-checkpoints-$LEACT_ROLE"
if [[ ! -e "$LEACT_LEGACY_SCRATCH" && ! -L "$LEACT_LEGACY_SCRATCH" ]]; then
    ln -s "$LEACT_CHECKPOINT_STAGING" "$LEACT_LEGACY_SCRATCH"
fi

phase=checking_runtime
python -m pytest reproductions/leact/tests -q
python reproductions/leact/cluster/preflight.py \
    --gpus "$LEACT_GPUS" --output "$LEACT_ROLE_ROOT/runtime.json"
timeout --signal=TERM --kill-after=30 180 \
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$LEACT_GPUS" \
    reproductions/leact/cluster/check_collectives.py

phase=reproduction
driver_exit=0
timeout --signal=TERM --kill-after=300 23h \
    python -m reproductions.leact.reproduction \
    --root "$LEACT_STUDY_ROOT" --role "$LEACT_ROLE" \
    --oracle "$LEACT_SOURCE/reproductions/leact/artifacts/leduc6r2s/oracle.jsonl" \
    --saved-experiment "$LEACT_SAVED_EXPERIMENT" \
    --prompt-root "$LEACT_PROMPT_ROOT" \
    --scratch "$LEACT_CHECKPOINT_STAGING" \
    --gpus "$LEACT_GPUS" --rounds 3 --epochs 3 --batch-size 32 || driver_exit=$?

phase=collecting_results
for attempt in $(seq 1 24); do
    [[ -f "$LEACT_ROLE_ROOT/collection-exported.json" ]] && break
    sleep 5
done
if [[ "$driver_exit" == 0 ]]; then
    phase=complete
else
    phase=reproduction_failed
fi
exit "$driver_exit"
