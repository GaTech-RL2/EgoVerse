# Hardware session guide — 2026-08-06 — FINAL checkpoints only

Supersedes `hw_session_0805_guide.md` (that one was written mid-training). All four runs
are now **complete**; every checkpoint below is the final one, verified on disk. Nothing
here is a moving target.

Repo root `/coc/flash7/czhang883/Documents/EgoVerse/`, `source emimic/bin/activate`.

## The four policies

| # | policy | corpus / task | image | depth | **final checkpoint** | final val |
|---|---|---|---|---|---|---|
| 1 | **h_rect** | human — full nav+pick+place | LUT colour **RGB** | — | `logs/RBY1_human_rect/human_rect_resnet_2k/checkpoints/epoch_epoch=1999.ckpt` | 0.0656 |
| 2 | **a3r_tel_colour** | teleop — near-table manip | LUT colour **RGB** | ✅ 224 | `logs/RBY1_adapt3r_tel_colour/adapt3r_tel_colour_rgb_2k/checkpoints/epoch_epoch=1999.ckpt` | 0.1394 |
| 3 | **a3r_human** | human — full task | LUT colour **RGB** | ✅ 224 | `logs/RBY1_adapt3r_human/adapt3r_human_2k/checkpoints/epoch_epoch=1999.ckpt` | 0.0903 |
| 4 | **dp3_tight1024** | teleop — near-table manip | — | ✅ cloud | `logs/RBY1_dp3_tight1024/dp3_tight1024_2k/checkpoints/epoch_epoch=1899.ckpt` | ~0.1223 |

⚠ **#4 has no ep1999.** Its run's last saved state is **ep1899** (verified:
`epoch=1899, step=190000`; `last.ckpt` is the same file). That IS its final checkpoint —
do not go looking for a 1999 that does not exist.

All four are **RGB on the wire**. The only BGR frame in the room is the raw camera image
going *into* the LUT helper, which flips it. Verify per policy at serve time.

---

## 0. Shared setup (once)

### 0a. Live depth (for #2, #3, #4)

Must be **slam-rect frame**: rectified LEFT SLAM camera, K = fx=fy=**307.336684**,
c=(256,256) @512×512, **metres** float32, **0 = invalid**. Any other frame invalidates
everything below.

**Sanity probe before the first rollout (2 min):** one live depth frame at the table —
median of valid pixels ≈1–2 m, table plane smooth, table-edge distance within ~5 cm of a
tape measure. Wrong depth makes all three depth policies fail identically and it looks
like bad policies.

### 0b. Raw image → LUT (for #1, #2, #3)

**CORRECTED 2026-08-17:** with the `lawrence_custom` profile the raw RGB shm frame is
**already 640×640 as delivered** — feed the LUT directly. (Robot Aria native calib =
2016×1512 f=877.96; on-device 640² = crop 1512² @x=252 × 640/1512 → predicts f=371.6
vs our solved 370.40, 0.3% agreement.) The earlier 2560×1920/crop@320 recipe here was
the HUMAN device's mode — wrong for the robot unit. Only if a full-res 2016×1512
frame ever arrives: crop 1512² at (x=252, y=0), then INTER_AREA to 640.

```python
import numpy as np, cv2
L  = np.load("ai_docs/assets_rect_lut/robot_rect224_lut.npz")
MX, MY = L["map_x"], L["map_y"]

def rect_rgb_224(raw_640_bgr):                # 640x640 BGR as delivered
    rect = cv2.remap(raw_640_bgr, MX, MY, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rect[..., ::-1].copy()             # -> RGB. DO NOT SKIP.
```
Self-check: save one output frame — basket **blue**, penguin feet **orange**.
LUT validated vs the depth-exact training pipeline at median 1.44 px @224.

### 0c. Depth image (for #2, #3)

```python
def depth_224(depth_512):
    return cv2.resize(depth_512, (224, 224), interpolation=cv2.INTER_NEAREST)
    # NEAREST only. Never bilinear. Never in-paint the zeros.
```

### 0d. Proprio / actions (all four)

```python
"robot0_joint_pos": (22,) float32   # no-wheel [torso 6, r_arm 7, l_arm 7, head 2]
"hand_left_qpos":   (12,) float32
"hand_right_qpos":  (12,) float32
```
Out: `(1, 32, 49)` @10 Hz. `base[0:3]` per-step deltas → cumsum in the frame-0 heading;
`torso[3:9] head[9:11] l_arm[11:18] r_arm[18:25] l_hand[25:37] r_hand[37:49]` absolute.

⚠ **If you validate predictions against logged actions:** the policy predicts
`actions.joint_base_torso_head_arm_hand`, **not** `actions.joint`. Both are 49-D, both
exist in every dataset, and they are *different vectors*. Scoring against the wrong one
makes a healthy policy look broken by ~3.7× (this cost us a debugging cycle).

Serve: `python egomimic/scripts/serve_policy.py --checkpoint <ckpt> --port 800X`.
Metadata on connect must read `embodiment=rby1, action_dim=49, action_horizon=32`.

---

## 1. h_rect — full task

```python
obs = {"front_img_1": rect_rgb_224(raw), "robot0_joint_pos": q22,
       "hand_left_qpos": hl, "hand_right_qpos": hr}
```
```bash
python egomimic/scripts/serve_policy.py --checkpoint logs/RBY1_human_rect/human_rect_resnet_2k/checkpoints/epoch_epoch=1999.ckpt --port 8000
python egomimic/scripts/test_serve_policy_client.py --dataset-folder datasets/human_fullpp_rgbd --episode-idx 0 --max-steps 30 --trajectory
```
- **Dry-run reference (final ckpt, ep0, 30 rows, horizon 32): full-chunk MAE 0.0113,
  t1 0.0114, arms+hands 0.0124.** Reproduce to ~1e-3 → serving is faithful. (Flow-matching
  sampling is stochastic; not bit-exact.)
- Full-task start (away from table). Human-corpus policy — near-table close-up starts are
  known-OOD for the whole family.
- Executes at ~1/6 speed (60 Hz rows labeled 10 Hz). Slow smooth motion is correct.
- Best offline human policy (0.0651 best / 0.0656 final) and its attention locks onto the
  penguin/basket even on unseen robot frames (entropy 0.58 — the most focused of the four).
- LUT border is 9.5% vs 4.1% in training; if behavior looks border-sensitive, that's the
  first suspect.

## 2. a3r_tel_colour — near-table manip

```python
obs = {"front_img_1": rect_rgb_224(raw), "aria_depth": depth_224(d512),
       "robot0_joint_pos": q22, "hand_left_qpos": hl, "hand_right_qpos": hr}
```
```bash
python egomimic/scripts/serve_policy.py --checkpoint logs/RBY1_adapt3r_tel_colour/adapt3r_tel_colour_rgb_2k/checkpoints/epoch_epoch=1999.ckpt --port 8001
python egomimic/scripts/test_serve_policy_client.py --dataset-folder datasets/rby1_teleop_colour_rgbd --episode-idx 0 --max-steps 30 --trajectory
```
- Near-table start (teleop data has no nav phase).
- Its encoder uses ROBOT intrinsics — zero device mismatch of the four.
- Offline context, now final: colour Adapt3R 0.1394 vs grey 0.1286 vs v4 RGB 0.0859 —
  colour did **not** help; this is the weakest offline policy in the set. Rolling it out
  tests whether the offline ordering holds on hardware.
- Attention sits on the grippers/table edges, not the objects (entropy 0.66).

## 3. a3r_human — full task ★ the depth A/B

```python
obs = same as #2
```
```bash
python egomimic/scripts/serve_policy.py --checkpoint logs/RBY1_adapt3r_human/adapt3r_human_2k/checkpoints/epoch_epoch=1999.ckpt --port 8002
python egomimic/scripts/test_serve_policy_client.py --dataset-folder datasets/human_fullpp_rgbd --episode-idx 0 --max-steps 30 --trajectory
```
- **Dry-run reference (final ckpt, same rows as #1): full-chunk MAE 0.0070, t1 0.0073,
  arms+hands 0.0082.**
- **Run this back-to-back with #1 from the same start pose.** Identical everything except
  depth, so the pair isolates depth's hardware contribution. Two pieces of evidence
  disagree about who wins, which is why hardware decides:
  - offline on human data: h_rect **better** by 0.025 (0.0656 vs 0.0903)
  - cross-embodiment probe on real robot obs: a3r_human **better** (0.204 vs 0.221
    full-chunk; 0.245 vs 0.265 arms+hands) — depth may transfer across the embodiment
    gap better than RGB features do
  - caveat: neither human policy beat the episode-mean baseline (0.121) on robot data
- Its encoder lifts with human intrinsics (f=308.05) vs live robot rect (307.34): 0.2%,
  negligible. Most diffuse attention of the set (entropy 0.78).

## 4. dp3_tight1024 — near-table manip, geometry only

**Distinct cloud recipe — do not reuse code from any glass-frame DP3.**

```python
u, v = np.meshgrid(np.arange(512), np.arange(512))
X = (u - 256.0) * depth_512 / 307.336684
Y = (v - 256.0) * depth_512 / 307.336684
rng = np.sqrt(X*X + Y*Y + depth_512**2)
m = (depth_512 > 0.05) & (rng > 0.25) & (rng < 1.5)      # RANGE crop, not Z-slab
if m.sum() < 64:
    m = (depth_512 > 0.05) & (rng < 3.0)                 # degenerate-frame fallback
P = np.stack([X[m], Y[m], depth_512[m]], 1).astype(np.float32)   # raster order — keep it
if len(P) > 16384:
    P = P[::len(P)//16384 + 1][:16384]                   # DETERMINISTIC stride
P = fps(P, 1024)                                         # egomimic...fps_pytorch
if len(P) < 1024:
    P = np.concatenate([P, P[np.random.randint(0, len(P), 1024 - len(P))]])
cloud = P @ R_device_rect.T                              # -> glass frame
```
```python
R_device_rect = np.array([
    [ 9.3490159428316122e-01, -1.4136695959716325e-01,  3.2553708197539882e-01],
    [ 1.4951092839033189e-01,  9.8876007316834003e-01,  1.4918621893400541e-16],
    [-3.2187806899300325e-01,  4.8671351351621289e-02,  9.4552927414170662e-01]])
obs = {"front_pcd_1": cloud,        # (1024,3) float32 — NOT uint8
       "robot0_joint_pos": q22, "hand_left_qpos": hl, "hand_right_qpos": hr}
```
```bash
python egomimic/scripts/serve_policy.py --checkpoint logs/RBY1_dp3_tight1024/dp3_tight1024_2k/checkpoints/epoch_epoch=1899.ckpt --port 8003
python egomimic/scripts/test_serve_policy_client.py --dataset-folder datasets/rby1_teleop_pcd1024_tight --episode-idx 0 --max-steps 30 --trajectory
```
- uint8 cloud → silently treated as an image (BGR-flip + /255). dtype matters.
- Its "attention" (max-pool critical points) sits on the table outline and arm
  silhouettes, barely on the objects — expect scene-geometry-driven behavior.
- Alone this reads as "does DP3 act sensibly on hardware", not as the tight-vs-glass A/B.
  For the A/B add `dp3_pcd1024_glass` ep1999 (Z-slab 0.25–2.0 + random subsample,
  `pcd_policy_deployment_guide.md` §1) as a fifth rollout.

---

## Checklist

1. `git pull`; `source emimic/bin/activate`.
2. Depth sanity probe (§0a).
3. One `rect_rgb_224` frame saved — blue-basket check (§0b).
4. Per policy: dry-run replay → compare to the reference MAE where given → serve → roll out.
5. Order: **#1 → #3 back-to-back, same start pose** (the depth A/B), then **#2 → #4** near-table.
6. Log per rollout: start pose, reached / grasped / placed, and any border or speed oddity.

## Offline standings (final, for expectation-setting)

| policy | final val | best val | note |
|---|---|---|---|
| h_rect | 0.0656 | 0.0651@1099 | best human-corpus policy to date |
| a3r_human | 0.0903 | 0.0895@699 | depth costs 0.025 in-domain |
| dp3_tight1024 | ~0.1223 | 0.1107@599 | tight crop lost to glass (0.1069) |
| a3r_tel_colour | 0.1394 | 0.1224@149 | colour did not beat grey (0.1286/0.1157) |
| *v4 RGB (reference)* | *0.0904* | *0.0859@299* | *the teleop RGB bar* |
