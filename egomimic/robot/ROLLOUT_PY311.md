# Python 3.11 rollout runtime

EgoMimic rollout and the ARX controller run in one Python 3.11 environment.
ROS Humble retains its system Python 3.10 for ROS-owned processes, but the live
rollout process must not import from `/opt/ros/humble/lib/python3.10`.

## Policy ownership

`egomimic.robot.rollout` owns only robot/replay execution. For policy rollout it
reads the checkpoint's saved Hydra `_target_`, loads that `Algo`, and calls its
`create_rollout_policy` method. Each algorithm module owns its live `Policy`:

- `algo/pi.py`: PI checkpoint preparation and PI inference behavior.
- `algo/hpt.py`: HPT inference and diffusion-step setup.
- `pipeline/algo.py`: the dependency-aware rollout graph and EVA codecs.
- `algo/act.py`: an explicit unsupported-live-EVA boundary until ACT defines a
  safe observation/action codec.

Shared code in `rollout/policy.py` is limited to the policy contract,
checkpoint dispatch, and legacy EVA observation/action adaptation. Embodiment
IDs are resolved through the canonical enum rather than duplicated in robot
code.

## Pinned Fold 28-to-100 checkpoint

The rollout artifact below is an immutable copy of the pre-consolidation Fold
Diffusion Policy checkpoint at epoch 3 / global step 133665. Its provenance is:

- Training config:
  `/coc/flash7/paphiwetsa3/experiments/rh_fold_speed28_20260821/source/egomimic_stack/egomimic/hydra_configs/experiment/fold/rh/dn_single_cart.yaml`
- Source checkpoint at capture time:
  `/coc/flash7/paphiwetsa3/experiments/rh_fold_speed28_20260821/source/egomimic_stack/logs/rh_fold_dp_wrist/dn_single_cart_h28to100_2026-08-21_21-47-00/checkpoints/last.ckpt`
- SHA-256:
  `f2052d92e0296e6d1a6bb7e2c1ddbc7756d5d192fe6ed9b7291d1eb547af211e`
- Immutable container source:
  `/home/robot/robot_ws/external_ckpts/fold28to100/dn_single_cart_h28to100_step133665_epoch3.source.ckpt`
- Stripped rollout artifact:
  `/home/robot/robot_ws/external_ckpts/fold28to100/dn_single_cart_h28to100_step133665_epoch3.rollout.ckpt`
- Rollout-artifact SHA-256:
  `94487041b570d9299f352dd12def55e4c3553f24605cbeea4fad3763e380de4e`

The training run can rewrite both `last.ckpt` and an epoch-named checkpoint
when it resumes within that epoch. Treat the step-numbered local copy plus its
SHA as immutable; do not rely on either run-directory alias remaining stable.
The compatibility loader is intentionally limited to this pre-consolidation
Pipeline schema: it restores the serialized Fold stage
classes needed for strict weight loading, without making the deleted legacy
Pipeline modules an active second stack. It retains both saved model branches
for strict loading, remaps the saved normalization IDs from EVA bimanual `8`
to canonical ID `6` and human bimanual `18` to canonical ID `3`, and exposes
only the EVA branch to live rollout.

The launcher bind-mounts the gitignored host `external_ckpts/` directory
read-only at `/home/robot/robot_ws/external_ckpts`, so checkpoints persist
across container recreation without being copied into the Docker image. Set
`EVA_CHECKPOINT_DIR` only when that host directory intentionally lives outside
the repository.

The checkpoint records a 100-step DDIM schedule and emits a 100-action chunk.
The rollout path strict-loads that original persistent schedule and all
335,564,596 model parameters, then installs a nonpersistent 16-step schedule
on the exact frozen Fold compatibility head. The checkpoint and 100-action
horizon remain unchanged. On Bonjour's RTX 4090, a no-hardware warm benchmark
measured median query times of 0.535 seconds at 100 steps and 0.0978 seconds at
16 steps, a 5.48x speedup. The 16-step query is still longer than one 33 ms
control period, so synchronous replanning can still cause a smaller periodic
rate dip; uninterrupted 30 Hz control requires a separate asynchronous design.

The artifact passed strict CUDA loading and a no-hardware synthetic
observation-to-command shadow test, producing a finite `(100, 20)` normalized
action chunk and finite 14-D decoded EVA command without initializing the
robot. Shape and finiteness alone do not validate the pose-frame contract, so
the maintained tests also exercise a nontrivial two-arm
hardware-to-training-to-hardware round trip. Run it from inside the live
container with:

```bash
python -m egomimic.robot.rollout \
  --arms both \
  --cartesian \
  --policy-path /home/robot/robot_ws/external_ckpts/fold28to100/dn_single_cart_h28to100_step133665_epoch3.rollout.ckpt \
  --query_frequency 30
```

Do not pass `--resampled-action-len`; this checkpoint already has its trained
100-action chunk. The shadow gate proves software compatibility and finite
decoding only; it is not a physical-motion safety acceptance. Reducing DDIM
steps changes the sampling trajectory, so the first 16-step deployment still
requires an attended behavior check with the physical E-stop available.

## EVA pose-frame contract

Fold Zarr conversion changes the orientation convention of every observed and
commanded EVA pose while leaving translation unchanged:

```text
R_dataset = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]] @ R_hardware
```

Live Pipeline rollout must apply this conversion before constructing
`state_ee_pose`. After unnormalization and wrist-relative-to-base reversion, it
must apply the transpose/inverse matrix to each arm's output orientation before
calling the hardware interface. Applying only one side is invalid; omitting
both can make rotations appear plausible while rotating relative translations
into the wrong base direction and conditioning the model outside its training
distribution. The shared source of truth is
`egomimic/rldb/embodiment/eva_frames.py`, which is also used by
`eva_to_zarr.py`. Do not duplicate this matrix in rollout code.

## Live safety gates

Live policy rollout requires CUDA unless `--allow-cpu-policy` is explicitly
passed for diagnostics. Controller construction does not home the robot. The
first intervention prompt authorizes homing, followed by one shadow inference
whose output is validated but never sent to the motors.

Every command then passes application-level checks before the ARX controller:
exact action shape, finite values, normalized gripper range, configured joint
limits, a joint delta bounded by the controller's 200 ms preview window, an
8 cm Cartesian translation bound, and a 0.5 rad Cartesian rotation bound. Any
exception or normal exit commands the measured joint state and stops camera
recorders. These are fail-closed rollout guards, not replacements for attended
hardware testing or the controller's internal limits.

Policy and teleoperation gripper values remain normalized: `0` is closed and
`1` is open. Only after `ARXInterface` denormalizes that command does closed
become the native `-0.012 m` X5A endpoint. Do not change a model's gripper
output range to include negative values. Running the Stanford
`calibrate.py --override-configs` path writes the nominal `0.0 m` close value
and erases this attended offset; restore and revalidate `-0.012 m` before
restarting collection or rollout.

## Laptop/offline setup

```bash
uv venv --python 3.11 emimic
source emimic/bin/activate
uv sync --active --frozen
python -c 'from egomimic.robot.backends.arx5 import optional_arx5_api; assert optional_arx5_api() is None'
pytest -q tests/test_arx5_backend.py tests/test_eva_frames.py tests/test_pipeline_rollout.py
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
./run_eva_docker.sh both
```

The launcher exposes the full USB bus so RealSense, Quest, and Aria devices can
re-enumerate without making a running container stale. It stores Aria pairing
and streaming certificates at the first configured location: the
`EVA_ARIA_AUTH_DIR` override, `$XDG_CONFIG_HOME/egomimic/aria`, or
`$HOME/.config/egomimic/aria`. That directory is mounted at `/root/.aria` in the
container and created with mode `0700`. The container is named
`egomimic-eva-live` by default; override it with `EVA_ROLLOUT_CONTAINER`. A
512 MiB private shared-memory allocation avoids the FastDDS shared-memory
fallback warning without exposing host IPC.

The build pins Stanford SDK commit
`a8890c9bae94464abd1cb7c5e4da7c4a62104a3a` and carries two reviewed local
patches: a 200 ms wait before reading the gripper calibration result, and an
X5-only `-0.012 m` command floor for the calibrated X5A grippers. The latter
derives its measured-state emergency floor from the same constant with the
upstream 5 mm tolerance (`-0.017 m`). Other Stanford model keys retain the
upstream `0 m` floor, and velocity limiting plus torque protection are
unchanged. Record the printed wheel SHA-256 with the deployed container digest.

Validate the built image without touching hardware:

```bash
docker run --rm --platform linux/amd64 --entrypoint /bin/bash \
  egomimic-eva:py311 -lc '
    python --version
    python -m pip check
    command -v adb
    python -c "import arx5_interface, aria.sdk, pyrealsense2; import egomimic.robot.rollout"
    pytest -q tests
  '
```

If Aria reports error 915 (`UsbNcmConnectionFailed`) while ADB and pairing still
work, fully restart the glasses rather than only reconnecting USB. This can
clear a stale device-side NetworkBoss service; no Aria network interface exists
on the host until that service successfully enables USB NCM.

The rollout and native ARX process are Python 3.11-only. ROS Humble on Ubuntu
22.04 retains its distribution-owned Python 3.10 for optional ROS helper nodes;
the rollout path imports the EVA interfaces directly and does not source the
ROS Python path. The `wsbuild` shell function runs the optional ROS build in a
subshell so those paths do not leak into the active rollout environment.

Do not install the wheel on macOS and do not command robot hardware from the
laptop. Hardware acceptance proceeds through import/CAN discovery, read-only
state, one-arm low-gain, gripper, bimanual, shadow-policy, then attended rollout
gates.

## Live acceptance checklist

Automated and read-only checks:

- Python 3.11, `pip check`, all tests, CUDA, and native ARX/Aria/RealSense
  imports.
- Both D405 serials produce frames.
- Container ADB selects the Quest rather than Aria, and receives finite left
  and right controller transforms plus buttons.
- Aria authentication/device info and a start-frame-stop RGB stream.
- Both CAN links are `ERROR-ACTIVE` with zero bus errors, followed by read-only
  state from each arm.

Attended hardware gates, in order:

- One-arm low-gain movement, then gripper and bimanual movement.
- Load the intended checkpoint and run shadow inference; validate the output
  without commanding motors.
- Run an attended policy rollout with the physical E-stop available.

On `bonjour` on 2026-08-23, all automated/read-only checks, the Quest input
path, an attended right-arm upward movement, gripper close, and bimanual
teleoperation passed. The first checkpoint rollout attempt exposed and was
stopped for a missing live EVA pose-convention boundary; the paired conversion
is now covered by offline regression tests. Attended policy rollout after
deploying that fix remains pending.
