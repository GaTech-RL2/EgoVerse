# ARX / EgoVerse Setup + Demo Collection

## 1. One-time host setup

### 1.1. udev rules

Create `/etc/udev/rules.d/99-eva.rules`:

```bash
# Right Arm
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="2077387F3430", SYMLINK+="eva_right_can"

# Left Arm
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="206634925741", SYMLINK+="eva_left_can"
```

Replace the `serial` values with your own device serial numbers, then reload:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 1.2. Bash alias for both arms

Add this to `~/.bashrc`:

```bash
alias can-both='  sudo pkill slcand;   sudo ip link delete can1 2>/dev/null;   sudo ip link delete can2 2>/dev/null;   sudo slcand -o -s8 /dev/eva_left_can can1 &   sudo slcand -o -s8 /dev/eva_right_can can2 &   sleep 0.5;   sudo ifconfig can1 up;   sudo ifconfig can2 up'
```

Then reload:

```bash
source ~/.bashrc
```

`can-both` will reset any existing CAN state, then bring up `can1` (left) and `can2` (right).

---

## 2. Docker image build (only when code changes or container doesn't start/attach)

From your EgoVerse repo:

```bash
cd path/to/your/EgoVerse/repo
git pull        # or `gt sync`
./scripts/build_arx5_py311_wheel.sh
docker build --platform linux/amd64 -t egomimic-eva:py311 .
```

You only need to do this initially and whenever you pull/modify code.

---

## 3. Running the container

From the repo root (or wherever `run_eva_docker.sh` lives):

```bash
./run_eva_docker.sh {left | right | both}
```

The launcher owns the foreground terminal and creates a disposable (`--rm`)
container. Keep that terminal open while using the robot; attach from a second
terminal or through Cursor / VS Code. The container disappears when its shell
exits, so rerun the launcher to create the next session.

The running container is named `egomimic-eva-live` by default. Attach through
Cursor / VS Code or from the host with:

```bash
docker exec -it egomimic-eva-live bash
```

The shell opens in `/home/robot/robot_ws` with the Python 3.11 environment
active. Rollout does not require a ROS workspace build. If you need the
optional ROS helper nodes, run `wsbuild`; it builds in an isolated subshell so
ROS Humble's Python 3.10 paths do not leak into the rollout shell.

---

## 4. Connect Aria + VR

### 4.1. Aria pairing (inside container)

With Aria connected to the companion app:

```bash
aria auth pair
```

Pairing persists on the host under `~/.config/egomimic/aria` by default, even
though the interactive container is disposable. Set `EVA_ARIA_AUTH_DIR` before
running `run_eva_docker.sh` to use a different host directory.

### 4.2. Check VR + Aria inside the container

Inside `egomimic-eva-live`:

```bash
adb start-server
adb devices
aria device info
```

---

## 5. Ensure arms are connected

On the host:

```bash
can-both
```

This brings up `can1` and `can2` for the left and right arms.

---

## 6. Collecting demos

> **Warning (hardware connections, before running `collect_demo.py`):**
> - Plug the dock into the **THUNDERBOLT (PCIe) port below the GPU**.
> - Plug the **Aria separately into a USB port** (not through the dock).

Inside the container, from `/home/robot/robot_ws`:

```bash
python -m egomimic.robot.collect_demo
```

Defaults:

- Saves demos to `./demos`
- Uses the **right** arm by default

### 6.1. Useful arguments

```bash
python -m egomimic.robot.collect_demo \
  --auto-episode-start {episode_idx} \
  --demo-dir /path/to/demo/directory \
  --arms {right | left | both} \
  --calibrate
```

- `--auto-episode-start {episode_idx}`: auto-increments episode index starting at `episode_idx`
- `--demo-dir`: custom demo output directory
- `--arms`: choose `right`, `left`, or `both`
- `--calibrate`: run Quest controller orientation calibration

### 6.2. Quick controls (Quest controller)

- **Y**: reset robot to home
- **B**: start / stop episode recording
- **X**: delete current episode buffer
- **A**: exit the collection loop. This is not a hardware E-stop; use the
  physical emergency stop for an unsafe robot condition.

- Left / right triggers: engage robot motion  
- Left / right front triggers: control gripper

### 6.3. Gripper close calibration

The direct Python collection and rollout paths keep each gripper command
normalized at the policy/application boundary: `0` is closed and `1` is open.
Only `ARXInterface` denormalizes that value into the `close` and `open`
endpoints in `eva_ws/src/config/configs.yaml`. Both X5A close endpoints are
currently `-0.012 m` because the calibrated zero does not fully close these
grippers. The application maps `1` to the calibrated YAML open endpoint, but
the native X5 controller clips that command to its `0.088 m` maximum. Never
emit `-0.012` from a policy; it is a native post-denormalization endpoint.

The pinned Stanford wheel carries an X5-only command floor of `-0.012 m` and a
measured-state emergency floor of `-0.017 m`, preserving 5 mm of feedback
tolerance. Its velocity and torque protections remain active. The YAML is read
once when `ARXInterface` starts, so restart collection after an edit. Edits
inside the disposable container are not persisted: change the host YAML,
rebuild the image, and recreate the container. A native-floor change also
requires rebuilding the wheel before the image. Changing YAML alone cannot
bypass the wheel's floor.

The legacy `calibrate.py --override-configs` path writes the nominal calibrated
close value `0.0 m` and therefore erases the attended `-0.012 m` close offset.
Do not use that option without restoring and revalidating the per-arm close
endpoint.

Validate one gripper at a time with the physical E-stop available before
bimanual teleoperation. The optional ROS helper is outside this acceptance
path: it reads the colcon install/share config, ignores these `gripper`
endpoints, imports the legacy binding, and applies its own fixed `-0.018 m`
offset.

---

## 7. Common errors

### 7.1. Resource busy (stale `collect_demo.py`)

If you see a “resource busy” error, inside the container:

```bash
jobs -l
kill -9 {pid_of_previous_collect_demo.py}
```

Then rerun:

```bash
python -m egomimic.robot.collect_demo
```

### 7.2. `ModuleNotFoundError: No module named 'arx5_interface'`

The live controller now uses the pinned CPython 3.11 wheel. Rebuild it and then
rebuild the robot image:

```bash
./scripts/build_arx5_py311_wheel.sh
docker build --platform linux/amd64 -t egomimic-eva:py311 .
```

See [`../ROLLOUT_PY311.md`](../ROLLOUT_PY311.md). Sourcing ROS does not install
the ARX binding and must not add ROS's Python 3.10 site-packages to rollout.

### 7.3. `CXXABI_1.3.15` / `libstdc++.so.6` error

Example:

```text
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
```

Rebuild the manylinux wheel using `scripts/build_arx5_py311_wheel.sh`. Do not
work around an ABI mismatch by exporting a Python 3.10 conda library path.

### 7.4. VR debug mode popup missing

Inside `egomimic-eva-live`:

```bash
adb kill-server
adb start-server
adb devices
```

### 7.5. Aria error 915 (`UsbNcmConnectionFailed`)

Fully restart the glasses; unplugging/reconnecting USB or repeating pairing is
not a full reboot. On `bonjour`, error 915 came from a stale device-side
NetworkBoss service even though ADB and authentication worked, and a full
glasses reboot restored USB streaming.

---

## 8. Uploading demos to AWS

After you are done collecting data:

```bash
python egomimic/scripts/data_upload/eva_uploader.py
```
