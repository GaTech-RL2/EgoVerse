# Python 3.11 rollout runtime

EgoMimic rollout and the ARX controller run in one Python 3.11 environment.
ROS Humble retains its system Python 3.10 for ROS-owned processes, but the live
rollout process must not import from `/opt/ros/humble/lib/python3.10`.

## Laptop/offline setup

```bash
uv venv --python 3.11 emimic
source emimic/bin/activate
uv sync --active --frozen
python -c 'from egomimic.robot.backends.arx5 import optional_arx5_api; assert optional_arx5_api() is None'
pytest -q tests/test_arx5_backend.py tests/test_pipeline_rollout.py
```

The native ARX wheel is intentionally absent on macOS. Offline rollout and all
observation/action codec tests still run there.

## Robot image build

Build the pinned, patched CPython 3.11 wheel with a Docker runtime that can
execute `linux/amd64` containers. This works on x86_64 Linux and on Apple
Silicon with Colima's VZ/Rosetta support:

```bash
./scripts/build_arx5_py311_wheel.sh
docker build --platform linux/amd64 -t egomimic-eva:py311 .
```

The build pins Stanford SDK commit
`a8890c9bae94464abd1cb7c5e4da7c4a62104a3a` and carries one reviewed local
patch: a 200 ms wait before reading the gripper calibration result. Record the
printed wheel SHA-256 with the deployed container digest.

Validate the built image without touching hardware:

```bash
docker run --rm --platform linux/amd64 --entrypoint /bin/bash \
  egomimic-eva:py311 -lc '
    python --version
    python -m pip check
    python -c "import arx5_interface, aria.sdk, pyrealsense2; import egomimic.robot.rollout"
    pytest -q tests
  '
```

The rollout and native ARX process are Python 3.11-only. ROS Humble on Ubuntu
22.04 retains its distribution-owned Python 3.10 for optional ROS helper nodes;
the rollout path imports the EVA interfaces directly and does not source the
ROS Python path.

Do not install the wheel on macOS and do not command robot hardware from the
laptop. Hardware acceptance proceeds through import/CAN discovery, read-only
state, one-arm low-gain, gripper, bimanual, shadow-policy, then attended rollout
gates.
