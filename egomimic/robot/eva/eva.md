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
docker build -t robot-env:latest .
```

You only need to do this initially and whenever you pull/modify code.

---

## 3. Running the container

From the repo root (or wherever `run_eva_docker.sh` lives):

```bash
./run_eva_docker.sh {left | right | both}
```

Once the container appears in Cursor / VS Code, attach a terminal to it and run:

```bash
cd /home/robot/robot_ws
wsbuild
cd ../..
```

---

## 4. Connect Aria + VR

### 4.1. Aria pairing (inside container)

With Aria connected to the companion app:

```bash
aria auth pair
```

### 4.2. Check VR + Aria from host

On the host machine:

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
python3 collect_demo.py
```

Defaults:

- Saves demos to `./demos`
- Uses the **right** arm by default

### 6.1. Useful arguments

```bash
python3 collect_demo.py   --auto-episode-start {episode_idx}   --demo-dir /path/to/demo/directory   --arms {right | left | both}   --calibrate
```

- `--auto-episode-start {episode_idx}`: auto-increments episode index starting at `episode_idx`
- `--demo-dir`: custom demo output directory
- `--arms`: choose `right`, `left`, or `both`
- `--calibrate`: run Quest Pro controller orientation calibration

### 6.2. Quick controls (Quest controller)

- **Y**: reset robot to home
- **B**: start / stop episode recording
- **X**: delete current episode buffer
- **A**: ESTOP

- Left / right triggers: engage robot motion  
- Left / right front triggers: control gripper

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
python3 collect_demo.py
```

### 7.2. `ModuleNotFoundError: No module named 'arx5'`

Inside the container:

```bash
source /opt/ros/humble/setup.bash
```

### 7.3. `CXXABI_1.3.15` / `libstdc++.so.6` error

Example:

```text
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
```

Fix:

```bash
export LD_LIBRARY_PATH=/root/.local/share/mamba/envs/arx-py310/lib:$LD_LIBRARY_PATH
```

### 7.4. VR debug mode popup missing (host)

On the host:

```bash
adb kill-server
adb start-server
adb devices
```

---

## 8. Uploading demos to AWS

After you are done collecting data:

```bash
python3 eva_uploader.py
```