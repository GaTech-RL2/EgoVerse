# RBY1 Policy Hardware Deployment — Operator Handoff

> **Audience:** whoever is pulling checkpoints onto the robot host, serving them,
> and running closed-loop rollouts on RBY1 + XHand + Aria.
>
> **Code roots (this machine):**
> - EgoVerse (serve / ckpts / LeRobot datasets): `~/RB_Y1_workspace/EgoVerse`
> - SEW-Geometric-Teleop (robot rollout / Aria / hands): `~/RB_Y1_workspace/SEW-Geometric-Teleop`
>
> **Branch:** EgoVerse `rby1_aria_policy` (latest) for unpickling recent ckpts.
>
> Companion docs:
> - Human / FPP nav+pick+place: `ai_docs/fpp_deployment_note.md`
> - Serving / debug basics: `ai_docs/deployment_plan.md`, `deployment_debug_guide.md`
> - SEW rollout internals: `SEW…/docs/ai/guides/policy_rollout_and_data.md`
> - Teleop data collection (SLAM/VIO): see Aria section below

---

## 0. Mental model (two processes)

```
┌─────────────────────────────┐         WebSocket :8000        ┌──────────────────────────────────┐
│  EgoVerse (emimic venv)     │ ◄──── obs payload ──────────── │  SEW rollout (conda env: rby1)   │
│  egomimic/scripts/           │ ──── actions (1,32,49) ──────► │  run_rollout_aria_egoposer.sh    │
│    serve_policy.py           │                               │  → rollout_rby1_policy_real.py   │
│  loads .ckpt + baked norms   │                               │  Aria RGB + HW proprio + XHand   │
└─────────────────────────────┘                               └──────────────────────────────────┘
```

1. **Terminal A — serve** a checkpoint (GPU, `emimic` venv).
2. **Terminal B — rollout** on hardware or MuJoCo sim (`rby1` conda). Never mix the two envs.

On connect, metadata **must** read: `embodiment=rby1`, `action_dim=49`, `action_horizon=32`.

---

## 1. Observation / action contract (all recent policies)

### Obs (client → server)

| key | shape / type | notes |
|---|---|---|
| `front_img_1` | `(224,224,3)` **numpy** uint8 **BGR** | cv2-native. Server flips BGR→RGB on the **numpy** path only. A torch tensor is **not** flipped → silent channel swap. |
| `robot0_joint_pos` | `(22,)` float | **No-wheel:** `position[4:26]` = torso6 + r_arm7 + l_arm7 + head2. Sending 26-D crashes. |
| `hand_left_qpos` / `hand_right_qpos` | `(12,)` each | Measured hand state. |

Server may list cosmetic extras (`task_id`, repeated camera keys) — harmless if the client is metadata-driven (our wrappers are).

### Action (server → client)

`(1, 32, 49)` un-normalized, ~10 Hz chunk. Layout:

| slice | meaning |
|---|---|
| `base[0:3]` | per-step **deltas** `(dx, dy, dyaw)` → integrate by **plain cumsum in frame-0 heading** (no yaw rotation of the xy deltas) |
| `torso[3:9]` | absolute |
| `head[9:11]` | absolute |
| `l_arm[11:18]` / `r_arm[18:25]` | absolute |
| `l_hand[25:37]` / `r_hand[37:49]` | absolute |

Wrapper preset: `--action-map-preset joint_base_torso_head_arm_hand`.

### Image path on hardware

Aria → publisher shared memory → **224² BGR policy crop** (center-crop + resize; no undistort) → sent as `front_img_1`. Do **not** send RGB.

---

## 2. Which policy family am I rolling?

| Family | What it is | Start pose | Typical dataset for GT / reset |
|---|---|---|---|
| **FPP / HD** (human data) | Nav + approach + manip; vision-driven after HD dropout-0.9 | From distance for nav tests; near-table was OOD for human-only policies | `datasets/aria_fullpp` |
| **Teleop v1 / v2 / v3** (robot demos) | **Manipulation-only** near-table fix; perception matches robot | **At / next to the table**, gripper near teleop start. Base motion is small coordinating motion, **not** a nav drive | `rby1_teleop_pp_0724` or `_sg` |

**Current recommended hardware ladder (teleop manip):**

1. **v3-A** `RBY1_wb_img_tel_v3/…/epoch_epoch=1999.ckpt` — try first (72 demos)
2. **v3-C** same dir `epoch_epoch=599.ckpt` — if 1999 feels brittle/over-tight
3. **v2** `RBY1_wb_img_tel_v2/…/epoch_epoch=1499.ckpt` — **known-good** on this robot
4. Round-1 ResNet (`tel_resnet@999`) — older baseline

FPP/HD human policies: see `fpp_deployment_note.md` (different start pose / expectations).

---

## 3. Pull checkpoints (robot host ← Skynet)

Scripts live in EgoVerse root. Password once via env (nothing written to disk):

```bash
cd ~/RB_Y1_workspace/EgoVerse
read -rs -p "Skynet pw: " SKYNET_PASS; export SKYNET_PASS; echo

# Teleop v3 (primary 1999 + fallback 599)
bash pull_teleop_pp_0724_v3.sh

# Teleop v2 (1499 + SG dataset)
bash pull_teleop_pp_0724_v2.sh

# Teleop round-1 A/B/C + dataset
bash pull_teleop_pp_0724.sh

# FPP HD-era A/B/C
bash pull_fpp_hd.sh
```

Local ckpt layout (examples):

```
checkpoints/RBY1_wb_img_tel_v3/wb_img_pickplace_v3_72demo_2k/checkpoints/epoch_epoch=1999.ckpt
checkpoints/RBY1_wb_img_tel_v2/wb_img_pickplace_v2_2k/checkpoints/epoch_epoch=1499.ckpt
checkpoints/rby1_teleop_pp_0724/tel_resnet_1k/checkpoints/epoch_epoch=999.ckpt
```

---

## 4. Serve

```bash
cd ~/RB_Y1_workspace/EgoVerse
source emimic/bin/activate

# Required deps (already on this host; re-check on a new machine):
#   msgpack-numpy, websockets

python egomimic/scripts/serve_policy.py \
  --checkpoint checkpoints/RBY1_wb_img_tel_v3/wb_img_pickplace_v3_72demo_2k/checkpoints/epoch_epoch=1999.ckpt \
  --port 8000
```

Verify on connect: **rby1 / 49 / 32**. Leave this terminal running. Do **not** include `serve_policy.py` in cleanup while rolling out.

Optional pre-flight against training corpus (EgoVerse):

```bash
python egomimic/scripts/test_serve_policy_client.py \
  --episode-idx 0 --max-steps 30 --trajectory \
  --dataset-folder datasets/rby1_teleop_pp_0724_sg
```

---

## 5. Sim first (GT obs), then hardware

Wrappers (SEW):

- Sim: `projects/rby1_teleop/run_rollout_aria_egoposer_sim.sh`
- Real: `projects/rby1_teleop/run_rollout_aria_egoposer.sh`

Env knobs (both):

| env | default | meaning |
|---|---|---|
| `PORT` | 8000 | must match serve |
| `DATASET` | … | LeRobot path for GT modes + `dataset_avg` soft-reset |
| `DEMO_NAME` | 0 | episode index for GT / viewer |
| `GT_MODE` | (empty) | `gt_proprio` / `gt_action` / … |
| `EXEC_STEPS` | 6 | first K of 32-step chunk executed |
| `FREQ` | 10 | Hz |
| `SAFE_MODE` | 1 (real) | y/N before each chunk |
| `SHOW_CAMERA` | 1 | live OpenCV of policy frame |
| `REC_COMPARE_DIR` | (empty) | if set: save `sent_*.png` + `live_*.png` |
| `FREEZE_HEAD` | 0 | hold initial look-down head |

### 5a. Sim GT (dataset image + proprio → policy → MuJoCo)

```bash
cd ~/RB_Y1_workspace/SEW-Geometric-Teleop
GT_MODE=gt_proprio PORT=8000 \
  DATASET=/home/aloha/RB_Y1_workspace/EgoVerse/datasets/rby1_teleop_pp_0724_sg \
  DEMO_NAME=0 \
  bash projects/rby1_teleop/run_rollout_aria_egoposer_sim.sh
```

PASS: smooth chunk overlay tracking the demo (training distribution).

### 5b. Hardware live

Prereqs:

```bash
# once per boot
sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1 card_label=AriaCam

# left hand EtherCAT NIC must exist (default enx4cd717ad4a22)
ip -br link show | grep enx
```

```bash
cd ~/RB_Y1_workspace/SEW-Geometric-Teleop
mkdir -p projects/collected_data/rollout_reviews

SAFE_MODE=1 SHOW_CAMERA=1 EXEC_STEPS=16 FREQ=10 PORT=8000 \
  DATASET=/home/aloha/RB_Y1_workspace/EgoVerse/datasets/rby1_teleop_pp_0724_sg \
  REC_COMPARE_DIR=projects/collected_data/rollout_reviews/tel_v3_1999_$(date +%Y%m%d_%H%M%S) \
  bash projects/rby1_teleop/run_rollout_aria_egoposer.sh
```

- Soft-resets to **dataset-average** start pose (confirm when prompted).
- Teleop policies: stand **at the table**, gripper near teleop start.
- E-stop ready. First motion is the soft reset.
- `--hold-on-stale-frame` is default-on in real rollout: if Aria freezes, base holds instead of acting on a stale image.

Without `REC_COMPARE_DIR`, **no video/frames are saved** (`SHOW_CAMERA` is display-only).

Encode a review MP4 later:

```bash
D=$(ls -td projects/collected_data/rollout_reviews/tel_v3_* | head -1)
ffmpeg -y -framerate 5 -pattern_type glob -i "$D/live_*.png" \
  -c:v libx264 -pix_fmt yuv420p "$D/live.mp4"
```

---

## 6. Cleanup (after failed starts / before a fresh run)

`Connection refused` on `:8000` = **server not running** (often killed by a previous `pkill serve_policy`). Robot/Aria/hands can still come up and then abort — clean those leftovers:

```bash
cd ~/RB_Y1_workspace/SEW-Geometric-Teleop
python stop_aria_streaming.py --force --ignore-errors

sudo pkill -f 'rollout_rby1_policy_real|rollout_rby1_policy_sim' 2>/dev/null || true
sudo pkill -f 'aria_v2_streaming_publisher|test_aria_streaming_gui' 2>/dev/null || true
sudo pkill -f 'demo_rby1_xr_robot_teleop' 2>/dev/null || true
sudo pkill -f 'xhand_replay_qpos|hand_replay_qpos' 2>/dev/null || true
# Only if you intend to stop the policy server too:
# sudo pkill -f 'serve_policy.py' 2>/dev/null || true

sudo rm -f /tmp/rby1_policy_real_*.lock
sudo rm -f /dev/shm/psm_* 2>/dev/null || true

pgrep -af 'rollout_rby1|aria_v2_streaming|serve_policy|xhand_replay' || echo clean
ss -ltnp | grep 8000 || echo "port 8000 free"
```

Aria-only stop (device + local `:6768`): `python stop_aria_streaming.py` at SEW root
(symlink also at `~/RB_Y1_workspace/shuo_proj/robot_constraint/stop_aria_streaming.py`).

Standalone Aria stream test (no robot):  
`conda run -n rby1 python projects/rby1_teleop/test_aria_streaming_gui.py`

---

## 7. Triage cheat sheet

| Symptom | Likely cause |
|---|---|
| `Connection refused` `:8000` | Server not running / wrong port |
| `size of tensor a (26) must match … (22)` | Sending raw joint_pos; need 22-D no-wheel `[4:26]` |
| Colors inverted / reaches wrong place | Sending torch / already-RGB instead of numpy BGR |
| Base drives away on manip policy | Delta→cumsum integration bug, not the policy |
| Grasp fails, arm OK | Expected weak spot (hand block largest error) |
| Nav-ish teleop policy stays put / near-table FPP confuses | Wrong family for the start pose |
| Aria freezes mid-run | USB strain; hold-on-stale engages; reseat cable / restart publisher |
| Left hand EtherCAT `Failed to open device` | `enx…` NIC missing (USB-Ethernet unplugged), not a rename |
| `modprobe: v4l2loopback is in use` | Kill old publisher, then reload module |

---

## 8. Teleop data collection (high-dim Aria obs) — for later offline stereo

Recorder: `demo_rby1_xr_robot_teleop_functional_wb_with_aria_info.py` with `--aria_streaming`.

Keys (HDF5 `obs/`):

- `aria_image` — **policy** BGR crop (not full FOV)
- `slam/<label>/image` — raw **fisheye** grayscale, native res, **no** rotate/undistort  
  labels: `slam-front-left/right`, `slam-side-left/right` (stereo = front pair)
- `slam/.../calib` JSON — intrinsics + `T_device_camera`
- `vio_pose` `(N,4,4)` `T_odometry_device`, `vio_ts`
- Timestamps are **host receive** time; rows are teleop-paced (frames often repeat — dedupe via `*_ts`)

Profile: `projects/rby1_teleop/lawrence_custom.json` (RGB/SLAM ~10 Hz, VIO ~20 Hz).

---

## 9. Envs reminder

| Task | Env |
|---|---|
| `serve_policy.py`, pull scripts, EgoVerse train/client | `source emimic/bin/activate` |
| Hardware / sim rollout, Aria publisher, XHand, `rby1_sdk` | `conda activate rby1` (wrappers do this) |
| EtherCAT left hand | rollout re-execs with `sudo -E` |

---

## 10. Quick copy-paste: current best teleop HW loop

```bash
# A — serve
cd ~/RB_Y1_workspace/EgoVerse && source emimic/bin/activate
python egomimic/scripts/serve_policy.py \
  --checkpoint checkpoints/RBY1_wb_img_tel_v3/wb_img_pickplace_v3_72demo_2k/checkpoints/epoch_epoch=1999.ckpt \
  --port 8000

# B — roll (near table)
cd ~/RB_Y1_workspace/SEW-Geometric-Teleop
SAFE_MODE=1 SHOW_CAMERA=1 EXEC_STEPS=16 FREQ=10 PORT=8000 \
  DATASET=/home/aloha/RB_Y1_workspace/EgoVerse/datasets/rby1_teleop_pp_0724_sg \
  REC_COMPARE_DIR=projects/collected_data/rollout_reviews/tel_v3_1999_$(date +%Y%m%d_%H%M%S) \
  bash projects/rby1_teleop/run_rollout_aria_egoposer.sh
```

Fallback ckpt path if v3 disappoints:  
`checkpoints/RBY1_wb_img_tel_v2/wb_img_pickplace_v2_2k/checkpoints/epoch_epoch=1499.ckpt`

---

*Last updated: 2026-08-02 — covers teleop v2/v3 pull scripts, SEW wrappers, Aria SLAM recording path, and HW cleanup lessons from robot-host sessions.*
