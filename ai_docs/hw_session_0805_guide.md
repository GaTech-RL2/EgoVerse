# Hardware session guide — 2026-08-05 — h_rect / a3r_tel_colour / a3r_human / dp3_tight1024

One self-contained guide for exactly these four rollouts. Everything here uses the
**LATEST checkpoint** (session rule). All commands from repo root
`/coc/flash7/czhang883/Documents/EgoVerse/` with `source emimic/bin/activate`.

| # | policy | corpus / task | image | depth | checkpoint |
|---|---|---|---|---|---|
| 1 | **h_rect** | human — full nav+pick+place | LUT colour **RGB** | — | `logs/RBY1_human_rect/human_rect_resnet_2k/checkpoints/last.ckpt` |
| 2 | **a3r_tel_colour_rgb** | teleop — near-table manip | LUT colour **RGB** | ✅ 224 | `logs/RBY1_adapt3r_tel_colour/adapt3r_tel_colour_rgb_2k/checkpoints/last.ckpt` |
| 3 | **a3r_human** | human — full task | LUT colour **RGB** | ✅ 224 | `logs/RBY1_adapt3r_human/adapt3r_human_2k/checkpoints/last.ckpt` |
| 4 | **dp3_tight1024** | teleop — near-table manip | — | ✅ cloud | `logs/RBY1_dp3_tight1024/dp3_tight1024_2k/checkpoints/last.ckpt` |

**No BGR policy in this set.** Every image below is RGB after the flip in the LUT
helper — the raw robot frame is BGR and the flip is mandatory (frozen-DINO policies
are silently degraded by swapped channels; we measured this the hard way).

**Checkpoint freshness:** #1 and #3 finish ep1999 tonight; #2 is mid-training
(~ep500+, improving hourly). `last.ckpt` is always valid to serve — just note the
epoch you grabbed (`ls -la` the file, or it's printed at serve start) so results are
attributable.

---

## 0. Shared components (set up once)

### 0a. Live depth assumption (applies to #2/#3/#4)

Depth must be **slam-rect frame** metric depth: rectified LEFT SLAM camera,
K = fx=fy=**307.336684**, c=(256,256) at 512×512, metres float32, **0 = invalid**.
If your live depth is not exactly this frame, nothing below is valid — stop and
re-warp first.

**2-minute sanity probe before any rollout:** capture one live depth frame pointed
at the table and check: median of valid pixels ≈ 1–2 m, table region smooth, no
scale surprise (table edge distance should match a tape measure within ~5 cm).
If depth is wrong, all three depth policies fail identically and it will look like
"bad policies".

### 0b. The LUT image warp (applies to #1/#2/#3)

Precomputed remap: raw 640×640 fisheye → 224×224 colour-rect policy frame.
Validated vs the depth-exact training pipeline: median 1.44 px @224.

```python
import numpy as np, cv2
L  = np.load("ai_docs/assets_rect_lut/robot_rect224_lut.npz")
MX, MY = L["map_x"], L["map_y"]

def rect_rgb_224(raw_bgr_640):                    # robot's raw 640x640 fisheye, BGR
    rect = cv2.remap(raw_bgr_640, MX, MY, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rect[..., ::-1].copy()                 # -> RGB. DO NOT SKIP.
```

Quick self-check at the robot: save one output frame — basket must look **blue**,
penguin feet **orange**. If they look orange/cyan respectively, the flip is missing.

### 0c. Depth image for the two Adapt3R policies (#2/#3)

```python
def depth_224(depth_512):                          # metres float32, rect frame
    return cv2.resize(depth_512, (224, 224), interpolation=cv2.INTER_NEAREST)
    # NEAREST only. Never bilinear (flying pixels), never in-paint zeros.
```
This is pixel-aligned to `rect_rgb_224`'s output by construction (same frame),
up to the LUT's 1.44 px residual.

### 0d. Proprio + action (identical for all four)

```python
"robot0_joint_pos": (22,) float32   # no-wheel order [torso 6, r_arm 7, l_arm 7, head 2]
"hand_left_qpos":   (12,) float32
"hand_right_qpos":  (12,) float32
```
Action out `(1, 32, 49)` un-normalized @10 Hz: `base[0:3]` per-step deltas →
integrate by plain **cumsum in the frame-0 heading**; all other blocks absolute
joint targets (`torso[3:9] head[9:11] l_arm[11:18] r_arm[18:25] l_hand[25:37]
r_hand[37:49]`).

Serving (each policy): `python egomimic/scripts/serve_policy.py --checkpoint <ckpt>
--port 800X`. Connect metadata must read `embodiment=rby1, action_dim=49,
action_horizon=32`. Adapt3R ckpts trigger a one-time `torch.hub` dinov2 warm-up in
`serve_policy.py` — first load is slower, that's normal.

---

## 1. h_rect — full task (human corpus; the new offline best, 0.0651)

```python
obs = {"front_img_1": rect_rgb_224(raw), "robot0_joint_pos": q22,
       "hand_left_qpos": hl, "hand_right_qpos": hr}
```
```bash
python egomimic/scripts/serve_policy.py --checkpoint logs/RBY1_human_rect/human_rect_resnet_2k/checkpoints/last.ckpt --port 8000
python egomimic/scripts/test_serve_policy_client.py --dataset-folder datasets/human_fullpp_rgbd --episode-idx 0 --max-steps 30 --trajectory   # dry run
```
- Task setup: the FULL task (start away from table, nav + pick + place) — this is a
  human-corpus policy; near-table close-up starts are known-OOD for the whole family.
- Human-data timing: executes human motion at ~1/6 speed (60 Hz rows labeled 10 Hz).
  Expect slow, smooth motion — that is correct behavior, not a bug.
- Known deltas vs training (details in `human_rect_deployment_note.md`): LUT border
  9.5% vs 4.1% in training (watch for border-sensitivity), device focal 0.2%.
- Reference to compare against: hd_wam3's prior full-task rollouts.

## 2. a3r_tel_colour_rgb — near-table manip (teleop corpus; colour+depth)

```python
obs = {"front_img_1": rect_rgb_224(raw), "aria_depth": depth_224(d512),
       "robot0_joint_pos": q22, "hand_left_qpos": hl, "hand_right_qpos": hr}
```
```bash
python egomimic/scripts/serve_policy.py --checkpoint logs/RBY1_adapt3r_tel_colour/adapt3r_tel_colour_rgb_2k/checkpoints/last.ckpt --port 8001
python egomimic/scripts/test_serve_policy_client.py --dataset-folder datasets/rby1_teleop_colour_rgbd --episode-idx 0 --max-steps 30 --trajectory
```
- Task setup: near-table manipulation start (teleop data contains no nav).
- Its encoder lifts depth with the ROBOT intrinsics (exact match to live) — of the
  two Adapt3Rs this one has zero device mismatch.
- Mid-training checkpoint: record the epoch. Offline it trails v2/v4-class RGB
  policies; the question this rollout answers is behavior, not offline parity.

## 3. a3r_human — full task (human corpus; the "does depth help on hardware" probe)

```python
obs = same as #2 (LUT RGB image + 224 depth + proprio)
```
```bash
python egomimic/scripts/serve_policy.py --checkpoint logs/RBY1_adapt3r_human/adapt3r_human_2k/checkpoints/last.ckpt --port 8002
python egomimic/scripts/test_serve_policy_client.py --dataset-folder datasets/human_fullpp_rgbd --episode-idx 0 --max-steps 30 --trajectory
```
- Full-task setup, 1/6-speed expectation — same as #1.
- Loses to h_rect offline by 0.024; hardware has flipped an offline verdict before
  (hd_wam3). **The informative comparison is #3 vs #1 back-to-back on the same
  scene/start** — that pair isolates depth's hardware contribution exactly.
- Device deltas: its encoder lift uses the HUMAN intrinsics baked into the ckpt
  (f=308.05) vs live robot rect (f=307.34) — 0.2%, negligible. Training depth was
  ≤66 ms stale (7.5 Hz nearest); fresher live depth is strictly in-distribution.

## 4. dp3_tight1024 — near-table manip (geometry-only; tight cloud recipe)

**Do NOT reuse any other cloud code — this recipe differs from the glass builds.**

```python
# depth_512: metres float32, rect frame
u, v = np.meshgrid(np.arange(512), np.arange(512))
X = (u - 256.0) * depth_512 / 307.336684
Y = (v - 256.0) * depth_512 / 307.336684
rng = np.sqrt(X*X + Y*Y + depth_512**2)
m = (depth_512 > 0.05) & (rng > 0.25) & (rng < 1.5)      # RANGE crop, not Z-slab
if m.sum() < 64:                                          # degenerate-frame fallback
    m = (depth_512 > 0.05) & (rng < 3.0)
P = np.stack([X[m], Y[m], depth_512[m]], 1).astype(np.float32)  # raster order — keep
if len(P) > 16384:
    P = P[::len(P)//16384 + 1][:16384]                    # DETERMINISTIC stride
P = fps(P, 1024)                                          # egomimic...fps_pytorch
if len(P) < 1024:
    P = np.concatenate([P, P[np.random.randint(0, len(P), 1024 - len(P))]])
cloud = P @ R_device_rect.T                               # glass frame; R = T_device_rect[:3,:3]
```
```python
obs = {"front_pcd_1": cloud,                # (1024,3) float32 — NOT uint8!
       "robot0_joint_pos": q22, "hand_left_qpos": hl, "hand_right_qpos": hr}
```
```bash
python egomimic/scripts/serve_policy.py --checkpoint logs/RBY1_dp3_tight1024/dp3_tight1024_2k/checkpoints/last.ckpt --port 8003
python egomimic/scripts/test_serve_policy_client.py --dataset-folder datasets/rby1_teleop_pcd1024_tight --episode-idx 0 --max-steps 30 --trajectory
```
- `T_device_rect` rotation: attrs of any depth-store file, or `rgbd_data_handoff.md` §1.
- uint8 cloud = silently treated as an image (BGR-flip + /255) — dtype matters.
- Note: without a glass-frame DP3 rollout in the same session, this reads as "does
  DP3 work on hardware at all", not as the tight-vs-glass A/B. If you want the A/B,
  add `dp3_pcd1024_glass@1999` (Z-slab 0.25–2.0 + random-subsample recipe,
  `pcd_policy_deployment_guide.md` §1) as a fifth rollout.

---

## Session checklist

1. `git pull` on `rby1_aria_policy`; `source emimic/bin/activate`.
2. Depth sanity probe (§0a) — before anything else.
3. One saved frame from `rect_rgb_224` — blue basket check (§0b).
4. Per policy: dry-run replay (commands above) → serve → rollout.
5. Suggested order: **#1 → #3 back-to-back** (same start pose; the depth A/B),
   then **#2 → #4** on the near-table setup.
6. Log per rollout: checkpoint epoch, start pose, and the failure mode if any
   (reached? grasped? placed? border-weirdness?). The h_rect border question (§1)
   and the a3r depth question (#3 vs #1) are the two things this session can
   uniquely answer.
