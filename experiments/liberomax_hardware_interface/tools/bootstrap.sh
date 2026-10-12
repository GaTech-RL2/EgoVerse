#!/usr/bin/env bash
set -Eeuo pipefail
set +x
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p /workspace/hardware-study/artifacts/runtime
cd /workspace/hardware-study
exec > >(tee artifacts/runtime/bootstrap.log) 2>&1
base64 --decode /tmp/hardware-payload.tar.gz.b64 > /tmp/hardware-payload.tar.gz
echo "$PAYLOAD_SHA256  /tmp/hardware-payload.tar.gz" | sha256sum --check --status
tar xzf /tmp/hardware-payload.tar.gz
python3 - <<'PY'
import json, os
from pathlib import Path
receipt = {'source_commit': os.environ['HARDWARE_SOURCE_COMMIT'],
           'payload_sha256': os.environ['PAYLOAD_SHA256'], 'source': os.environ['MAX_SOURCE'],
           'stage': os.environ['HARDWARE_STAGE'], 'source_materialized': True, 'model_calls': 0}
Path('artifacts/runtime/input-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
PY
if [[ "$HARDWARE_STAGE" == staging-check ]]; then exit 0; fi
archive_on_exit() {
    local result=$?
    if [[ "$result" != 0 ]]; then
        for log in artifacts/runtime/*install.log artifacts/runtime/*tests.log; do
            if [[ -f "$log" ]]; then tail -n 50 "$log"; fi
        done
    fi
    python -m experiments.liberomax_hardware_interface.tools.archive || { if [[ "$result" == 0 ]]; then result=1; fi; }
    exit "$result"
}
trap archive_on_exit EXIT
cat > /tmp/hardware-apt.list <<'APT'
deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20251001T000000Z/ bookworm main
deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/20251001T000000Z/ bookworm-security main
APT
APT_OPTIONS=(-o Dir::Etc::sourcelist=/tmp/hardware-apt.list -o Dir::Etc::sourceparts=- -o Acquire::Retries=3)
apt-get "${APT_OPTIONS[@]}" update -qq > artifacts/runtime/apt-install.log 2>&1
apt-get "${APT_OPTIONS[@]}" install -y --no-install-recommends git cmake g++ libgl1 libegl1 libglib2.0-0 libosmesa6 libseccomp2 libmagickwand-dev ffmpeg >> artifacts/runtime/apt-install.log 2>&1
python3 -m venv emimic
source emimic/bin/activate
python -m pip install pip==24.3.1 setuptools==75.8.0 wheel==0.45.1
python -m pip install torch==2.7.1+cpu torchvision==0.22.1+cpu --index-url https://download.pytorch.org/whl/cpu > artifacts/runtime/dependency-install.log 2>&1
python -m pip install -r experiments/liberomax_hardware_interface/requirements.txt >> artifacts/runtime/dependency-install.log 2>&1
# robosuite 1.4.0 imports termcolor without declaring it as a dependency.
python -c 'import robosuite, mujoco; assert robosuite.__version__ == "1.4.0"; assert mujoco.__version__ == "3.2.6"'
fetch_source() {
    git init -q "$1"
    git -C "$1" remote add origin "$2"
    git -C "$1" fetch --depth 1 origin "$3"
    git -C "$1" checkout --detach FETCH_HEAD
    test "$(git -C "$1" rev-parse HEAD)" = "$3"
}
fetch_source upstream-max https://github.com/liberomax/LIBERO-MAX.git 01625e290b1aa07511bdc820781e635e851353cc
if [[ "$MAX_SOURCE" == plus ]]; then
    fetch_source upstream-source https://github.com/sylvestf/LIBERO-plus.git 4976dc30028e805ff8094b55501d532c48fec182
    python -m experiments.liberomax_hardware_interface.tools.prepare_plus_assets --source-root upstream-source --receipt artifacts/runtime/plus-assets.json
    python - <<'PY'
import json
from pathlib import Path
source=Path('upstream-source/libero/libero').resolve()
config=Path('source-config'); config.mkdir()
paths={'benchmark_root':source, 'bddl_files':source/'bddl_files', 'init_states':source/'init_files',
       'assets':source/'assets', 'datasets':Path('unused-demonstrations').resolve()}
(config/'config.yaml').write_text(json.dumps({k:str(v) for k,v in paths.items()}))
PY
elif [[ "$MAX_SOURCE" == pro ]]; then
    fetch_source upstream-source https://github.com/Zxy-MLlab/LIBERO-PRO.git 2b910b5b5f53016bef9907632f6f840f1ce2229c
    # The pinned upstream setup repeats --include; hf keeps only the last list
    # (README.md). Fetch every required pattern in a single list first, then run
    # the unchanged upstream path checks and source-lock writer.
    hf download zhouxueyang/LIBERO-Pro --repo-type dataset \
        --revision c86fc3b8293185a6f373677018ff3e37f8391602 \
        --include 'bddl_files/**' 'init_files/**' 'metadata/**' README.md \
        --local-dir pro-substrate
    python upstream-max/scripts/setup_libero_pro_substrate.py --libero-pro-root upstream-source --dataset-root pro-substrate --config-dir source-config
else
    exit 2
fi
export LIBERO_SOURCE_PACKAGE_ROOT="$PWD/upstream-source/libero"
export LIBERO_CONFIG_PATH="$PWD/source-config"
export PYTHONPATH="$PWD/upstream-max/scripts/libero_source_overlay:$PWD/upstream-max/src:$PWD"
python -m pip freeze --all > artifacts/runtime/pip-freeze.txt
python -m libero_max validate-manifest upstream-max/benchmark/max8000/libero_max_8000.json > artifacts/runtime/max-validation.json
python -m libero_max validate-manifest upstream-max/benchmark/lite/libero_max_lite.json > artifacts/runtime/lite-validation.json
(cd upstream-max && PYTHONPATH="$PWD/src:$PWD" python -m pytest -q > ../artifacts/runtime/upstream-tests.log 2>&1)
python -m pytest --confcutdir=tests/unit/hardware_interface tests/unit/hardware_interface -q --junitxml=artifacts/runtime/unit-tests.xml > artifacts/runtime/unit-tests.log 2>&1
# Import the actual benchmark package and load every selected reset before
# requesting simulator commissioning. Unit fixtures do not import this stack.
python - <<'PY'
import json, os
from pathlib import Path
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero_max.substrate import load_case_task
from astra_reversal.hardware_interface.liberomax import commissioning_cases, load_release
cases=commissioning_cases(load_release(Path('upstream-max'), 'lite'), os.environ['MAX_SOURCE'])
for case in cases:
    task, states=load_case_task(case, benchmark)
    bddl=Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    if not bddl.is_file() or not len(states) > case['init_state_index']:
        raise ValueError('selected_benchmark_reset_unavailable')
receipt={'runtime_imports_passed':True, 'selected_resets_loaded':len(cases),
         'source':os.environ['MAX_SOURCE'], 'model_calls':0, 'scored':False}
Path('artifacts/runtime/import-check.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt), flush=True)
PY
if [[ "$HARDWARE_STAGE" == runtime-check ]]; then exit 0; fi
nvidia-smi --query-gpu=name,driver_version --format=csv > artifacts/runtime/gpu.txt
python -m experiments.liberomax_hardware_interface.tools.commission --max-root upstream-max --source-root upstream-source --source "$MAX_SOURCE" --out artifacts/commission
