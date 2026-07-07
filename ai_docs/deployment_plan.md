# Deployment Plan — RBY1 Whole-Body Image Policy

> **For the downstream deployment agent.** This deploys the trained policy
> (`aria_egoposer`, whole-body 49-D action from real Aria image + proprio) on the
> real RBY1. It runs as **two processes in two environments**:
>
> 1. **Policy server** — *this* repo + venv, on the GPU box. Loads the `.ckpt`,
>    serves over WebSocket + msgpack. (`egomimic/scripts/serve_policy.py`)
> 2. **Deployment receiver** — a *separate* env on the robot side. Each control
>    step it sends an observation and receives an action chunk, then executes it.
>
> The server is embodiment-agnostic plumbing; all robot-specific code lives in the
> receiver. Reference impls: `egomimic/serving/egoverse_policy.py`,
> `egomimic/serving/API.md`, `egomimic/scripts/test_serve_policy_client.py`.

```
 ROBOT SIDE (separate env)                          GPU BOX (this repo + emimic venv)
 ┌─────────────────────────┐   obs (msgpack)        ┌──────────────────────────────┐
 │ deployment_receiver.py  │ ───────────────────►   │ serve_policy.py              │
 │  - read camera (BGR)    │                        │  EgoVersePolicy(ModelWrapper)│
 │  - read joints (22-D)   │   actions (1,32,49)    │  - normalize → forward_eval  │
 │  - exec action chunk    │ ◄───────────────────   │  - UNnormalize → return      │
 └─────────────────────────┘   ws://host:8000       └──────────────────────────────┘
```

---

## 1. The contract (authoritative — read from `last.ckpt`)

**Endpoint:** `ws://<server-host>:8000`, payloads are `msgpack_numpy`. On connect the
server pushes a **metadata** dict; then the receiver sends obs dicts and gets action
dicts. HTTP `GET /healthz` → 200.

### Observation the receiver MUST send (per step)
| key | shape | dtype | notes |
|---|---|---|---|
| `front_img_1` | `(224, 224, 3)` | `uint8` | **BGR** (OpenCV order). Server converts BGR→RGB, `/255`, →NCHW. Match training res (224×224). Send **once**. |
| `robot0_joint_pos` | `(22,)` | float32 | **NO-WHEEL.** = the robot's 26-D joint state with the **first 4 base/wheel dims dropped** (`[4:26]`). Order must match training: `torso(6), right_arm(7), left_arm(7), head(2)`. |
| `hand_left_qpos` | `(12,)` | float32 | left hand joint pos |
| `hand_right_qpos` | `(12,)` | float32 | right hand joint pos |

> The baked schematic also lists `task_id` and extra action keys (an artifact of the
> config merge). They have no model input/stem and are ignored — **do not send
> `task_id`**; the four keys above are the entire input.

### Action the server returns
```json
{"actions": <float32 (1, 32, 49)>, "embodiment": "rby1", "server_timing": {...}}
```
- **Already un-normalized** (server applies `unnormalize_data`; receiver does NOT normalize/denormalize).
- Horizon 32 @ 10 Hz = 3.2 s chunk. 49-D layout (per `CLAUDE.md`; verified correct via
  proprio correlation — torso then head):

| block | idx | dim | meaning |
|---|---|---|---|
| base | `[0:3]` | 3 | **delta** x, y, yaw (per-step) — integrate |
| torso | `[3:9]` | 6 | torso joint targets |
| head | `[9:11]` | 2 | head joint targets |
| l_arm | `[11:18]` | 7 | left arm joint targets |
| r_arm | `[18:25]` | 7 | right arm joint targets |
| l_hand | `[25:37]` | 12 | left hand joint targets |
| r_hand | `[37:49]` | 12 | right hand joint targets |

**Gotchas:** 22-D no-wheel proprio (not 26); BGR uint8 image; output already
unnormalized; base block is per-step deltas; serving forces `num_inference_steps=10`.

---

## 2. Serve the policy (GPU box, this repo)
```bash
cd /coc/flash7/czhang883/Documents/EgoVerse && source emimic/bin/activate
export XDG_CACHE_HOME=/coc/flash7/czhang883/.cache HF_HOME=/coc/flash7/czhang883/.cache/huggingface
# vanilla head:
python egomimic/scripts/serve_policy.py --checkpoint logs/aria_egoposer/vanilla/checkpoints/last.ckpt --port 8000
# hierarchical head (use a different port to A/B):
python egomimic/scripts/serve_policy.py --checkpoint logs/aria_egoposer/hier/checkpoints/last.ckpt --port 8001
```
Confirm the metadata on connect matches §1 before doing anything else.

---

## 3. Receiver skeleton (robot side, separate env)
Separate venv needs only: `websockets`/`websocket-client`, `msgpack`, `msgpack_numpy`,
`numpy`, `opencv-python` (NOT torch/egomimic). Copy the client framing from
`egomimic/scripts/test_serve_policy_client.py:create_websocket_client` (≈ lines 140-164).

```python
# pseudo — see test_serve_policy_client.py for exact msgpack framing
ws = connect("ws://SERVER:8000"); meta = unpack(ws.recv())     # read contract
while running:
    obs = {
        "front_img_1":      cam_bgr_224,                       # (224,224,3) uint8 BGR
        "robot0_joint_pos": robot_joint_26[4:26].astype("f4"), # 22-D no-wheel
        "hand_left_qpos":   hl.astype("f4"),                   # 12
        "hand_right_qpos":  hr.astype("f4"),                   # 12
    }
    ws.send(pack(obs)); resp = unpack(ws.recv())
    act = resp["actions"][0]                                   # (32, 49) unnormalized
    execute(act[:K])                                           # receding horizon, K≈4-8
```

---

## 4. Debug ladder (run in order — only advance when the rung looks right)

The idea: swap **ground-truth (dataset) inputs** for **real (robot) inputs one
modality at a time**, so you can localize any sim→real gap (vision vs proprio).

| Rung | Proprio | Image | Where | Goal / PASS |
|---|---|---|---|---|
| **0. Plumbing** | GT | GT | offline, no robot/server | `test_serve_policy_client.py --local --trajectory` vs `datasets/aria_egoposer`. Validates ckpt + norm + 49-D layout. PASS: pred≈GT (low MAE), curves track. |
| **1. Wire** | GT | GT | server + receiver | Same GT inputs but through ws+msgpack. Validates the receiver/protocol. PASS: matches Rung 0. |
| **2. Real vision** | GT | **REAL** | robot + server | Feed recorded proprio but the **live camera**. Isolates the **vision** gap (real cam vs Aria). PASS: actions stay sensible. |
| **3. Real proprio** | **REAL** | GT | robot + server | Feed **live 22-D no-wheel** proprio but a recorded image. Isolates the **proprio** gap (ordering/units/offsets). PASS: actions sensible. |
| **4. Both real, open-loop** | REAL | REAL | robot, **do NOT execute** | Full real inputs; only **log + visualize** predicted actions (dry run). Confirms the whole input path before motion. |
| **5. Closed-loop** | REAL | REAL | robot, execute | Receding-horizon execution with safety limits. |

### Extra debug modes
- **Proprio-only / vision-only ablation:** at any rung, send a **black** `front_img_1`
  (zeros) to see if the policy still acts from proprio alone (it was trained with
  proprio-dropout 0.4/0.7, so it leans on vision — expect degraded but non-random);
  conversely zero the proprio to gauge vision-only. Brackets each modality's weight.
- **Whole-body action visualization:** for every rung, plot the predicted `(32,49)`
  chunk **by block** (base/torso/head/l_arm/r_arm/l_hand/r_hand) over the horizon, and
  offline overlay **pred vs GT** (reuse `test_serve_policy_client.py` plots /
  `visualize_lerobot_dataset.py`). Optionally render arm/hand joints on a MuJoCo/mink
  viewer to eyeball feasibility before executing.

---

## 5. Executing the action on RBY1
- Split the 49-D vector by the block table (§1). **base[0:3] is per-step delta**
  (dx,dy,dyaw) — integrate into a base command; the rest are **joint position targets**.
- **Receding horizon:** execute the first K≈4–8 of 32 steps, then re-query (don't run
  the whole 3.2 s chunk open-loop).
- vanilla = flat head, hier = DAG-decoded; identical I/O contract — A/B by port.

## 6. Safety
Dry-run (Rung 4) first; clamp joint pos/vel to limits; start from a safe home pose;
keep an e-stop; cap base deltas; ramp into closed-loop slowly.

## 7. ⚠️ Must-verify before Rung 3+ (robot-specific)
1. **Joint ordering match.** Training proprio/action assume RBY1 joints as
   `base(4 dropped), torso(6), right_arm(7), left_arm(7), head(2)` (proprio) and the
   §1 action layout. Confirm the **physical robot's** joint indices match — a different
   order silently breaks proprio + action mapping.
2. **Hand qpos convention** (12-D/hand) matches the teleop/retarget convention.
3. **Camera framing.** Real camera FOV/intrinsics differ from Aria; crop/resize to
   224×224 to match the training image distribution (Rung 2 will expose this).
4. Units (rad vs deg): training actions/proprio are radians; ensure the robot API agrees.

---

## 8. RETRAINED V1 policies

### 8.0 ⭐ ROUND 3 (2026-07-04) — `crop100_2k` — DEPLOY THIS ONE

Trained with crop ~0-100 px, proprio dropout 0.9, **per-joint raw-space noise σ=0.03 rad**
and a **±3 clamp on normalized proprio** (baked into the ckpt; fixes the proprio cliff
caused by near-static joints whose quantile range ~0 explodes small raw offsets).

| condition | round-2 crop_2k | **crop100_2k** | dino100_2k |
|---|---|---|---|
| clean | 0.012 | **0.013** | 0.025 |
| image shift 10/20 px | 0.011/0.010 | **0.012/0.013 (flat)** | 0.025/0.024 (flat) |
| proprio zeroed | 0.015 | **0.016** | 0.157 ✗ |
| proprio noise σ=0.01→0.05 rad | **0.234 (cliff)** | **0.014 at all σ ✓** | 0.023–0.028 ✓ |

- **Primary: `logs/aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt` — send REAL live
  proprio** (the §1 contract, no zeroing needed). Robust to framing shifts, to ≥3° of
  proprio error, AND to zeroed proprio. No special handling.
- A/B backup: `logs/aria_egoposer_firm/dino100_2k/checkpoints/last.ckpt` (frozen DINOv2)
  — noise/shift robust but ~2x looser clean and REQUIRES proprio (do not zero it).
- Image contract unchanged: raw fisheye 224x224 uint8 BGR, no rectification.

### 8.1 ROUND 2 (2026-07-03) — superseded by 8.0, kept for reference

Two 2000-epoch retrains of V1 on `aria_egoposer_firm` (RandomResizedCrop aug ~0-50 px,
joint-proprio dropout 0.8). Same §1 obs/action contract. Measured error profile
(MAE rad vs GT, 10 frames; "shift" = image translated then resized back to 224):

| condition | old vanilla_1k | **crop_2k** | **dino_2k** |
|---|---|---|---|
| clean | 0.009 | 0.012 | 0.013 |
| image shift 10/20 px | 0.019 / 0.035 | **0.011 / 0.010 (flat)** | 0.014 / 0.013 (flat) |
| proprio zeroed | 0.017 | **0.015** | 0.158 |
| proprio noise σ=0.01/0.02/0.03/0.05 rad | — / — / — / 0.139 | 0.234 at ALL σ ⚠️ | 0.039 / 0.047 / 0.070 / 0.112 |

- `crop_2k` = `logs/aria_egoposer_firm/v1_crop_2k/checkpoints/last.ckpt` (ResNet-18 + crop aug)
- `dino_2k` = `logs/aria_egoposer_firm/v1_dino_2k/checkpoints/last.ckpt` (frozen DINOv2 ViT-S/14)

**Deploy guidance from the profile:**
1. **Primary: serve `crop_2k` VISION-ONLY — send all-zero proprio vectors**
   (`robot0_joint_pos=zeros(22)`, `hand_*_qpos=zeros(12)`). It runs at ~clean accuracy
   (0.015) with zeros, is immune to 10-20 px framing shifts, BUT has a proprio cliff:
   even σ=0.01 rad (~0.6°) of proprio error collapses it to 0.234. Do NOT feed it live
   proprio unless the live values match training conventions exactly.
2. **Secondary A/B: `dino_2k` WITH live proprio** — degrades gracefully with proprio
   error (usable to ~0.02-0.03 rad / ~1-1.7°), but REQUIRES proprio (zeros → 0.158).
   Use it to test whether live proprio is good: if dino_2k works with live proprio but
   crop_2k doesn't, live proprio has small offsets; if neither works, look elsewhere.
3. Image contract unchanged: raw fisheye 224x224 square, uint8 BGR, no rectification.

## 9. V2 variant (DISCARDED) — image + 10-step joint history + 100-step base trajectory

There are **two trained vanilla policies** on the clean `_firm` data (both 49-D action, 32-step
horizon, identical output contract; in-dist MAE ~0.53° each):

| variant | checkpoint | extra obs vs V1 |
|---|---|---|
| **V1** (§1 contract) | `logs/aria_egoposer_firm/vanilla/checkpoints/last.ckpt` | — |
| **V2** | `logs/aria_egoposer_firm_v2/v2_hist_traj/checkpoints/last.ckpt` | joint **history** + base **trajectory** |

Serve V2 the same way (different port to A/B against V1):
```bash
python egomimic/scripts/serve_policy.py --checkpoint logs/aria_egoposer_firm_v2/v2_hist_traj/checkpoints/last.ckpt --port 8001
```
The **output** contract is identical to §1 (`(1,32,49)`, unnormalized, same block layout, base `[0:3]`
per-step deltas). Only the **input** changes: V2 replaces the single-step joint proprio with a
10-step history and adds a 100-step base trajectory.

### 8.1 V2 observation the receiver MUST send (per step)
| key | shape | dtype | notes |
|---|---|---|---|
| `front_img_1` | `(224,224,3)` | uint8 | **BGR**, raw fisheye 224² — same as V1 |
| `robot0_joint_pos_hist` | `(220,)` | float32 | last **10** no-wheel joint states (22-D each), **flattened oldest→newest, current last** |
| `base_traj` | `(300,)` | float32 | last **100** integrated base poses `[x,y,yaw]`, **first-frame world frame**, flattened oldest→newest, current last |
| `hand_left_qpos` | `(12,)` | float32 | same as V1 |
| `hand_right_qpos` | `(12,)` | float32 | same as V1 |

> Do **not** send `robot0_joint_pos` (single-step) for V2 — it's replaced by the history. The baked
> schematic still lists stale keys (`robot0_joint_pos`, `task_id`); the server **skips any key you
> don't send** (`if key in obs`), so just send the five above. Both windows are **flat vectors**
> (the server does `reshape(1,1,-1)` and normalizes with the baked stats — send raw values).

### 8.2 Copy-pasteable builders (maintain the two windows in the receiver)
```python
import numpy as np

class JointHistTracker:
    """V2 `robot0_joint_pos_hist`: last 10 no-wheel joint states (22-D), flattened
    oldest->newest (current last) -> 220-D. Causal left-pad with the first sample."""
    def __init__(self, steps=10, dim=22):
        self.steps, self.dim, self.buf = steps, dim, []
    def update(self, joint22):                 # call each control frame with the CURRENT 22-D no-wheel joint
        self.buf.append(np.asarray(joint22, np.float32).reshape(self.dim))
    def feature(self):
        b = self.buf[-self.steps:]
        b = [b[0]] * (self.steps - len(b)) + b
        return np.concatenate(b).astype(np.float32)      # (220,)

class BaseTrajTracker:
    """V2 `base_traj`: last 100 integrated base poses (x,y,yaw) in the rollout's FIRST-FRAME
    world frame, flattened oldest->newest (current last) -> 300-D. Built by PLAIN CUMSUM of the
    base deltas the policy commanded (matches training = cumsum of the demo's action base deltas).
    Do NOT rotate deltas by the running yaw (that body-frame integration was the export bug)."""
    def __init__(self, window=100):
        self.window = window
        self.cum = np.zeros(3, np.float32)               # x,y,yaw relative to frame 0
        self.buf = [self.cum.copy()]                     # frame-0 pose = [0,0,0]
    def commit_delta(self, base_delta):        # call after executing each control frame
        self.cum = self.cum + np.asarray(base_delta, np.float32)   # [dx,dy,dyaw]
        self.buf.append(self.cum.copy())
    def feature(self):
        b = self.buf[-self.window:]
        b = [b[0]] * (self.window - len(b)) + b
        return np.concatenate(b).astype(np.float32)      # (300,)
```

### 8.3 Receiver loop (V2)
```python
jh, bt = JointHistTracker(), BaseTrajTracker()          # reset at the START of each rollout
while running:
    jh.update(robot_joint_26[4:26])                     # measured 22-D no-wheel, current
    obs = {
        "front_img_1":           cam_bgr_224,           # (224,224,3) uint8 BGR raw fisheye
        "robot0_joint_pos_hist": jh.feature(),          # (220,)
        "base_traj":             bt.feature(),          # (300,)
        "hand_left_qpos":        hl.astype("f4"),       # (12,)
        "hand_right_qpos":       hr.astype("f4"),       # (12,)
    }
    chunk = infer(obs)["actions"][0]                    # (32,49) unnormalized
    for k in range(K):                                  # receding horizon, K~4-8
        execute(chunk[k])                               # joint targets + base delta chunk[k,0:3]
        bt.commit_delta(chunk[k, 0:3])                  # advance base_traj by the commanded delta
        if k < K - 1:
            jh.update(robot_joint_26[4:26])             # keep joint history at control rate
```

### 8.4 Conventions & gotchas (V2)
- **First-frame frame:** `base_traj` is anchored at the rollout's **first control frame** (origin, so
  `base_traj` is all-zeros on the first query). Reset both trackers at each rollout start.
- **Plain cumsum, not body-frame rotation:** integrate base deltas by simple accumulation in the
  frame-0 frame. Rotating each delta by the running yaw warps the XY path (yaw stays right) — that was
  the exact export bug the retarget agent fixed; don't reintroduce it on the deploy side.
- **Yaw is unwrapped/cumulative** (training summed raw `dyaw`, values reach ~1.5 rad). If you instead
  build `base_traj` from **measured odometry**, transform each pose into the first-frame frame and
  accumulate yaw with unwrapping — but commanded-delta cumsum (above) matches training exactly and is
  simplest; prefer it unless base tracking is poor.
- **Validated against the training columns** (`episode_000000`): `JointHistTracker` reproduces
  `obs.robot0_joint_pos_hist` **exactly** (1.000). For `base_traj`, committing the **next-frame**
  delta after each query (`deltas[t+1]`) reproduces `obs.base_traj` **exactly**; committing the
  current-frame delta (`chunk[k,0:3]`, as in the §8.3 loop) is a **constant ≤1-frame lag**
  (non-accumulating, ≤2.4 cm / ≤0.09 rad, because `deltas[0]=0`) — in-distribution and fine. For
  byte-exact `base_traj`, commit `chunk[k+1,0:3]` instead. (Alternatively, if the base exposes
  odometry, append the measured pose transformed into the first-frame SE(2) frame with unwrapped
  yaw — exact and robust to slip, but verify the odom frame/yaw-sign convention.)
- **Order:** flatten oldest→newest with the current sample **last** (both windows).
- Everything else (image BGR/raw-fisheye/224², 49-D output layout, receding horizon, safety) is per §1–§6.
