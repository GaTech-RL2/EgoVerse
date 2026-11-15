#!/usr/bin/env bash
set -Eeuo pipefail
set +x
trap 'code=$?; if [[ "$code" != 0 ]]; then sleep 45; fi; exit "$code"' EXIT
[[ "${LEACT_WORKFLOW_ID:?}" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "${LEACT_PAYLOAD_SHA256:?}" =~ ^[a-f0-9]{64}$ ]]
[[ "${BUILDER_ROLE:?}" == train || "$BUILDER_ROLE" == evaluate ]]
LEACT_REMOTE_BASE=/mnt/amlfs-07/shared/egoverse/rpunamiya/leact
LEACT_RUN_ROOT="$LEACT_REMOTE_BASE/runs/$LEACT_WORKFLOW_ID"
LEACT_SOURCE="$LEACT_RUN_ROOT/source"
BUILDER_STUDY=/mnt/amlfs-03/home/rpunamiya/leact/studies/builderbench-gpt6-student-20261024-v1
BUILDER_SHARED_ENV="$LEACT_REMOTE_BASE/runs/leact-leduc-qwen3-8b-7/source/emimic"
mountpoint -q /mnt/amlfs-03/home/rpunamiya
mkdir -p "$LEACT_RUN_ROOT" "$BUILDER_STUDY"
exec > >(tee -a "$LEACT_RUN_ROOT/bootstrap.log") 2>&1
ready=0
for attempt in $(seq 1 240); do
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
ln -s "$BUILDER_SHARED_ENV" "$LEACT_SOURCE/emimic"
cd "$LEACT_SOURCE"
source emimic/bin/activate
export PYTHONPATH="$LEACT_SOURCE"
export TMPDIR="/tmp/leact-builder-$BUILDER_ROLE"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
mkdir -p "$TMPDIR"
python - <<'PY'
import json, hashlib
from pathlib import Path
manifest=json.loads(Path('payload-manifest.json').read_text())
for name, spec in manifest['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==spec['sha256'], name
print('BUILDER_PAYLOAD_VERIFIED', flush=True)
PY
python -m pytest reproductions/leact/tests/test_builderbench.py reproductions/leact/tests/test_builderbench_speed.py reproductions/leact/tests/test_builderbench_student.py -q
python reproductions/leact/cluster/preflight.py --gpus 8 --output "$BUILDER_STUDY/runtime-$BUILDER_ROLE.json"
timeout --signal=TERM --kill-after=30 180 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 reproductions/leact/cluster/check_collectives.py
(
  flock -x 9
  if [[ ! -f "$BUILDER_STUDY/simulator-ready.json" ]]; then
    python -m uv venv --python "$BUILDER_SHARED_ENV/bin/python" "$BUILDER_STUDY/simulator"
    python -m uv pip install --python "$BUILDER_STUDY/simulator/bin/python" \
      'mujoco==3.17.0' 'dm-control==1.0.50' 'gymnasium==1.4.0' 'numpy==2.5.4' 'scipy==1.18.1'
    "$BUILDER_STUDY/simulator/bin/python" -c 'import json; from importlib.metadata import version; print(json.dumps({n:version(n) for n in ["mujoco","dm-control","gymnasium","numpy","scipy"]}))' > "$BUILDER_STUDY/simulator-ready.json"
  fi
) 9>"$BUILDER_STUDY/simulator-install.lock"
"$BUILDER_STUDY/simulator/bin/python" -m reproductions.leact.builderbench.simulator \
  --reference "$LEACT_SOURCE/reference" --replay "$LEACT_SOURCE/study-input/simulator-replays.json" \
  > "$BUILDER_STUDY/simulator-replay-$BUILDER_ROLE.json"
BUILDER_BASE=$(python -c 'import json; print(json.load(open("study-input/launch.json"))["base_model"])')
python -u -m reproductions.leact.builderbench.run_student --role "$BUILDER_ROLE" \
  --root "$BUILDER_STUDY" --base "$BUILDER_BASE" --corpus "$LEACT_SOURCE/study-input/corpus" \
  --reference "$LEACT_SOURCE/reference" --simulator-python "$BUILDER_STUDY/simulator/bin/python"
for attempt in $(seq 1 60); do
  [[ -f "$BUILDER_STUDY/collection-exported-$BUILDER_ROLE.json" ]] && break
  sleep 5
done
