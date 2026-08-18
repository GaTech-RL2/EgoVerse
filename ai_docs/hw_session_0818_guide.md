# HW session guide — 0818 — the three frame-transformed human policies

For the hardware-machine agent. Three NEW human-only policies, all trained with the
observation re-expressed into robot-native frames — the whole point is that the live
robot can construct these observations EXACTLY (its own FK is ground truth for the
transforms that had to be *estimated* on the training side). LATEST checkpoints only.

| # | policy | obs | val best (27-ep family gate) | run dir |
|---|---|---|---|---|
| 1 | **dp3_hglass** | 1024-pt cloud, glass frame, rect-Z crop | 0.0825 | `logs/RBY1_dp3_human_glass/dp3_human_glass_2k` |
| 2 | **dp3_eefball** | 1024-pt cloud, glass frame, **1.5 m ball around right eef** | 0.0828 | `logs/RBY1_dp3_eefball/dp3_eefball_2k` |
| 3 | **a3r_eef** | LUT RGB + depth + **per-frame eef extrinsic** | 0.0840 | `logs/RBY1_adapt3r_human_eef/adapt3r_human_eef_2k` |

Context: all three beat the camera-frame a3r_human (0.0903); h_rect (0.0651) remains
the RGB bar. The two DP3s statistically tie — the eef-ball needs nothing beyond arm's
reach, which is the more deploy-robust observation. Checkpoint = highest
`epoch_epoch=*.ckpt` in each run's `checkpoints/` (runs finish ~today; `last.ckpt`
equivalent).

---

## 0. Download / sync (from any Skynet login node, e.g. sky2.cc.gatech.edu)

```bash
# 1) CODE — MANDATORY, not optional: commit 13fe12f7 on rby1_aria_policy adds the
#    encoder file (previously untracked — old checkouts CANNOT unpickle these ckpts),
#    the per-frame-extrinsics routing, and a serving guard that raises if eef_T is
#    missing (instead of silently running the wrong frame).
cd ~/RB_Y1_workspace/EgoVerse && git pull   # must land >= 13fe12f7

# 2) CHECKPOINTS (grab the highest epoch_epoch=*.ckpt of each; ~1 GB apiece)
R=czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse
rsync -avP $R/logs/RBY1_dp3_human_glass/dp3_human_glass_2k/checkpoints/    ckpts/dp3_hglass/
rsync -avP $R/logs/RBY1_dp3_eefball/dp3_eefball_2k/checkpoints/            ckpts/dp3_eefball/
rsync -avP $R/logs/RBY1_adapt3r_human_eef/adapt3r_human_eef_2k/checkpoints/ ckpts/a3r_eef/

# 3) LUT + dry-run reference numbers (small)
rsync -avP $R/ai_docs/assets_rect_lut/ assets_rect_lut/
#    assets_rect_lut/dryref_new3.txt = per-policy reference MAEs on the FINAL ckpts
#    (auto-generated after training completes; if absent, wait for it)

# 4) OPTIONAL datasets for local replay dry-runs (recommended: the two DP3 ones)
rsync -avP $R/datasets/human_dp3_robotglass/  datasets/human_dp3_robotglass/   # 838M
rsync -avP $R/datasets/human_dp3_eefball/     datasets/human_dp3_eefball/      # 838M
# rsync -avP $R/datasets/human_fullpp_rgbd_eef/ datasets/human_fullpp_rgbd_eef/ # 14G, only if replaying a3r_eef locally
```

---

## 1. Constants (everything the pipeline needs, verbatim)

```python
import numpy as np
# Rect-frame pinhole (rectified LEFT SLAM), depth lives here. At 512x512:
FX = FY = 307.336684; CX = CY = 256.0          # robot device

# glass(device) <- rect rotation (pure rotation, zero translation).
R_DEV_RECT = np.array([
    [ 0.934902, -0.141367,  0.325537],
    [ 0.149511,  0.988760,  0.0     ],
    [-0.321878,  0.048671,  0.945529]])
T_DEV_RECT = np.eye(4); T_DEV_RECT[:3,:3] = R_DEV_RECT

# Solved Aria head mount, capture day 0726:  M = T_head2_device
# (link_head_2 -> Aria device/glass frame; from solve_head_mount.py, holdout-passed)
M_HEAD_DEV = np.array([
    [ 0.302994, -0.368514,  0.878859,  0.061702],
    [-0.952990, -0.119187,  0.278575,  0.048268],
    [ 0.002090, -0.921950, -0.387303,  0.121014],
    [ 0.0     ,  0.0     ,  0.0     ,  1.0     ]])
```

⚠ **Mount seating**: the datasets were built with the 0726 mount. If the glasses have
been re-seated since, re-run `solve_head_mount.py` (aria_gen2_scripts) and use the new
M in the FK chain below — a 2 cm mount error moves every eef-relative point by 2 cm.

### The FK chain (robot side — this is EXACT for you, estimated only in training)

```python
# From your own robot state (you already have FK in the SEW stack; the training side
# used MuJoCo model_v1.3_xhand_act.xml and certified this exact chain to 0.59 mm):
T_base_head2 = fk(q)["link_head_2"]     # base -> head link (torso 6 + head 2 joints)
T_base_eefR  = fk(q)["right_eef"]       # base -> right eef body
T_base_glass = T_base_head2 @ M_HEAD_DEV          # glass == Aria device frame
T_base_rect  = T_base_glass @ T_DEV_RECT          # rect frame pose in base
```
Everything below is built from these four transforms + live rect depth.

---

## 2. Perception pipelines, per policy

### 2a. dp3_hglass — the hardware-validated teleop recipe, unchanged

Identical to `pcd_policy_deployment_guide.md` §1 (the contract teleop DP3 already ran
on hardware). Zero new code:

```python
# depth_512: (512,512) float32 METRES, rect frame, 0=invalid
u,v = np.meshgrid(np.arange(512), np.arange(512))
X = (u-CX)*depth_512/FX; Y = (v-CY)*depth_512/FY; Z = depth_512
m = (Z>0.25) & (Z<2.0)                                   # Z-slab in RECT frame
P = np.stack([X[m],Y[m],Z[m]],1).astype(np.float32)
if len(P)>16384: P = P[np.random.choice(len(P),16384,replace=False)]
P = fps(P,1024)                                          # egomimic...fps_pytorch
cloud = P @ R_DEV_RECT.T                                 # -> glass frame
obs["front_pcd_1"] = cloud                               # (1024,3) float32 — NOT uint8
```

### 2b. dp3_eefball — same lift, crop = 1.5 m ball around the right eef

```python
Pr = np.stack([X[Z>0.05],Y[Z>0.05],Z[Z>0.05]],1).astype(np.float32)
Pg = Pr @ R_DEV_RECT.T                                   # all valid points -> glass
p_eef = (np.linalg.inv(T_base_glass) @ T_base_eefR)[:3,3]   # right eef IN GLASS frame
d = np.linalg.norm(Pg - p_eef, axis=1)
m = d < 1.5
if m.sum() < 64: m = d < 2.5                             # training fallback (never fired in 53,664 rows)
P = Pg[m]
if len(P)>16384: P = P[np.random.choice(len(P),16384,replace=False)]
obs["front_pcd_1"] = fps(P,1024)
```
Self-check before rollouts: max pairwise distance within any cloud must be ≤3.0 m
(training measured max 2.49). If you see 4+ m clouds, the eef transform is wrong.

### 2c. a3r_eef — LUT image + depth + the per-frame eef extrinsic

```python
L = np.load("assets_rect_lut/robot_rect224_lut.npz"); MX,MY = L["map_x"],L["map_y"]
rect = cv2.remap(raw_640_bgr, MX, MY, cv2.INTER_LINEAR, borderValue=0)  # raw frame IS 640x640
obs["front_img_1"] = rect[...,::-1].copy()               # -> RGB. mandatory.
obs["aria_depth"]  = cv2.resize(depth_512,(224,224),interpolation=cv2.INTER_NEAREST)

# THE NEW PIECE: E = T_eefR_rect (maps rect-frame points into the right-eef frame)
T_eef_rect = np.linalg.inv(T_base_eefR) @ T_base_rect
obs["eef_T"] = T_eef_rect.astype(np.float32).ravel()     # (16,), row-major
```
The updated serving **raises** if `eef_T` is missing — by design. If you see that
error, the extrinsic isn't wired; do not "fix" it by removing the key.

Sanity check on `eef_T` before trusting it (2 min): `inv(T_eef_rect)[:3,3]` is the eef
position in the rect camera frame — project it with (FX·x/z+CX, FY·y/z+CY)/512×224 and
the dot must land on the robot's own right gripper in the rect image. Training values
for scale: eef↔camera distance median 0.55 m.

### Shared

```python
obs["robot0_joint_pos"] = q22        # [torso 6, r_arm 7, l_arm 7, head 2]
obs["hand_left_qpos"]; obs["hand_right_qpos"]           # 12 each
```
Actions out `(1,32,49)` @10 Hz; base[0:3] deltas → cumsum in frame-0 heading; rest
absolute. **~1/6-speed replay is correct behavior** (human 60 Hz rows labeled 10 Hz).
Validate predictions against `actions.joint_base_torso_head_arm_hand`, never
`actions.joint`.

---

## 3. Verification ladder (climb in order)

1. **Serve + replay, zero hardware risk** — per policy:
   ```bash
   python egomimic/scripts/serve_policy.py --checkpoint <highest-epoch ckpt> --port 800X
   python egomimic/scripts/test_serve_policy_client.py \
     --dataset-folder datasets/<matching dataset> --episode-idx 0 --max-steps 30 --trajectory
   ```
   Compare to `assets_rect_lut/dryref_new3.txt` (reference MAE + literal first action
   vector per policy, computed on the final ckpts through this exact path). Match to
   ~1e-3 → your serving is faithful.
2. **Depth pre-flight** — tape-measure 3 points, 3% tolerance; median 1–2 m at the
   table; NEAREST resize only; 0 stays 0.
3. **Cloud QC live** — one frame each recipe; check against training stats:
   glass: range med≈1.9 p95≈2.5; eefball: diameter ≤3.0, range med≈1.3.
   `eef_T` projection check (§2c).
4. **Static obs, no motion** — ≥10 Hz inference, arm targets near current pose.
5. **Motion** — reduced speed, e-stop in hand, full-task start (away from the table;
   near-table close-ups are OOD for the human family). 5 rollouts each, log the
   furthest stage reached (approach / reach / contact / grasp / lift / place).

## 4. What to compare on hardware

The scientific pairings this session can uniquely answer:
- **dp3_hglass vs dp3_eefball** (same frame, only crop differs): does the tight,
  scene-independent observation transfer better on the real robot? Offline they tie.
- **a3r_eef vs the old a3r_human rollout expectations**: the eef frame is Adapt3R's
  actual cross-embodiment mechanism — this is its first hardware test.
- Any of these vs **h_rect** (RGB bar, `hw_session_0806_guide.md` §1) on the same
  start poses.

*Everything transform-side was gated before training: eef chain 2.7 cm holdout vs
on-device handtracking; FK chain certified 0.59 mm; eef-ball verified by true cloud
diameter; serving path smoke-tested end-to-end incl. the missing-eef_T guard
(job 3701492). Compiled 2026-08-18.*
