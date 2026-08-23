# HW session guide — 0823 — the dual / colour / transplant fleet

For the hardware-machine agent. Evaluates the policies NOT yet rolled out (the 0818
session covered dp3_hglass, dp3_eefball, a3r_eef). All checkpoints are FINAL ep1999,
verified on disk. Builds on `hw_session_0818_guide.md` §0–1 for shared constants
(K, R_DEV_RECT, M_HEAD_DEV, FK chain, depth pre-flight) — read that first; this guide
adds the new obs recipes.

## Priorities (a realistic session is the top 3)

| pri | policy | final ckpt (val best/final) | what its rollout answers |
|---|---|---|---|
| **1** | **dp3c_dual** | `logs/RBY1_dp3c_dual/dp3c_dual_2k/checkpoints/epoch_epoch=1999.ckpt` (0.0793/0.0827) | offline champion — does per-point colour survive the human→robot shift on hardware? |
| **2** | **dp3_dual** | `logs/RBY1_dp3_dual/dp3_dual_2k/checkpoints/epoch_epoch=1999.ckpt` (0.0804/0.0860) | the colourless dual — direct HW A/B vs dp3c (colour) and vs 0818's single-stream hglass |
| **3** | **dp3_transplant** | `logs/RBY1_dp3_transplant/dp3_transplant_2k/checkpoints/epoch_epoch=1999.ckpt` (0.0806/0.0849) | ★ offline-invisible test: trained on robot-arm clouds — ONLY hardware can score it vs dp3_dual |
| 4 | dp3_full_eefframe | `logs/RBY1_dp3_full_eefframe/.../epoch_epoch=1999.ckpt` (0.0809/0.0946) | best single-stream — full scene in eef coords |
| 5 | dp3_eefframe | `logs/RBY1_dp3_eefframe/.../epoch_epoch=1999.ckpt` (0.0824/0.0884) | eef-ball in eef coords |
| opt | dual_noprop / dual_pos3 / dual_eefonly | `logs/RBY1_dp3_dual_{noprop,pos3,eefonly}/...=1999.ckpt` | proprio ablations, only if the day runs long |

**The session's headline experiment is #2 vs #3 back-to-back from identical start
poses.** They are the same architecture and observation recipe — the ONLY difference
is that #3 trained on clouds where human arms were replaced by robot arms. Offline
they tie by construction (val clouds contain human arms); the robot's own clouds
contain robot arms, so hardware is the only judge. If #3 > #2, the embodiment gap in
the clouds was real and we know how to close it.

---

## 1. New robot-side transforms (on top of 0818 §1)

```python
# From 0818: T_base_head2, T_base_eefR via YOUR FK;  T_base_glass = T_base_head2 @ M_HEAD_DEV
T_glass_eefR = np.linalg.inv(T_base_glass) @ T_base_eefR      # right eef in glass
p_eefR_glass = T_glass_eefR[:3,3]
# ALSO needed for dp3c colour neutralization:
T_glass_eefL = np.linalg.inv(T_base_glass) @ T_base_eefL      # left eef in glass

# eef-pose proprio (dual policies): position + first two rotation columns (rot6d)
eef_pose_glass = np.concatenate([T_glass_eefR[:3,3],
                                 T_glass_eefR[:3,0], T_glass_eefR[:3,1]]).astype(np.float32)  # (9,)
```

## 2. Obs recipes per policy

### 2a. dp3_dual and dp3_transplant — IDENTICAL contract (that's the point)

```python
# depth_512 -> lift -> glass, as in 0818 §2a
Pg = lift_all_valid(depth_512) @ R_DEV_RECT.T            # all Z>0.05 points, glass frame

# stream 1: GLOBAL (the hardware-validated hglass recipe, unchanged)
Zr = (Pg @ R_DEV_RECT)[:,2]
P1 = Pg[(Zr>0.25)&(Zr<2.0)]
P1 = fps(sub16384(P1), 1024)                             # (1024,3) float32

# stream 2: LOCAL — 1.5 m ball around right eef, RE-EXPRESSED in eef coords
d  = np.linalg.norm(Pg - p_eefR_glass, axis=1)
P2 = Pg[d<1.5]                                           # fallback d<2.5 if <64 pts
P2 = fps(sub16384(P2), 1024)
P2 = P2 @ np.linalg.inv(T_glass_eefR)[:3,:3].T + np.linalg.inv(T_glass_eefR)[:3,3]

obs = {"front_pcd_1": P1, "front_pcd_2": P2,             # both (1024,3) float32
       "eef_pose_glass": eef_pose_glass,                 # (9,)
       "robot0_joint_pos": q22, "hand_left_qpos": hl, "hand_right_qpos": hr}
```
**dp3_transplant needs NOTHING extra at deploy** — no excision, no insertion. Its
training clouds were edited to contain robot arms precisely so the live cloud (which
naturally contains them) matches. Deploy identically to dp3_dual.

Self-checks: P2 max norm ≤ 3.0 m (ball property); project `p_eefR_glass` through
rect→image — dot must sit on the right gripper.

### 2b. dp3c_dual — coloured clouds (xyzrgb)

Build the cloud from the **224 grid** (this is how training built them):

```python
d224 = cv2.resize(depth_512,(224,224),interpolation=cv2.INTER_NEAREST)
rgb  = rect_rgb_224(raw_640_bgr).astype(np.float32)/255.0     # LUT (0818 §0b) -> RGB in [0,1]
# lift the 224 grid: K224: fx=fy=307.336684*224/512=134.46, cx=cy=98.0
m = d224>0.05
P = lift(d224[m]);  C = rgb[m]
C[C.sum(1)<0.02] = 0.5                                   # colour-invalid (LUT border) -> grey
Pg = P @ R_DEV_RECT.T
# EMBODIMENT COLOUR-EXCISION (mandatory — training had it):
near = (dist(Pg,p_eefR_glass)<0.30)|(dist(Pg,T_glass_eefL[:3,3])<0.30)
C[near] = 0.5
# streams: SAME crops as 2a, but carry colour through every index op
P1,C1 = crop_slab(Pg,C) -> sub16384 -> fps1024           # keep (points,colour) paired!
P2,C2 = ball_crop -> sub16384 -> fps1024 -> to_eef_coords(P2 only; colour unchanged)
obs["front_pcd_1"] = np.concatenate([P1,C1],1).astype(np.float32)   # (1024,6)
obs["front_pcd_2"] = np.concatenate([P2,C2],1).astype(np.float32)   # (1024,6)
# + eef_pose_glass + proprio as in 2a
```
⚠ Colour is **RGB in [0,1]** float — not BGR, not 0-255. Self-check: histogram the
blue channel of basket-region points — must dominate. Grey-out check: points within
0.30 m of either eef must be exactly 0.5.

### 2c. dp3_full_eefframe / dp3_eefframe — single stream in eef coords

```python
# full_eefframe: slab crop (2a stream-1) then re-express in eef coords
# eefframe:      ball crop (2a stream-2) — same thing, it IS stream 2
obs = {"front_pcd_1": P_eefcoords,                        # (1024,3)
       "robot0_joint_pos": q22, "hand_left_qpos": hl, "hand_right_qpos": hr}
```
(No eef_pose proprio, no second stream for these two.)

---

## 3. Verification ladder

1. **Replay dry-run per policy** (zero hardware risk) — serve + `test_serve_policy_client.py
   --dataset-folder datasets/<matching> --episode-idx 0 --max-steps 30 --trajectory`.
   Reference numbers (final ckpts, real serving path): **`ai_docs/assets_rect_lut/dryref_0823.txt`**
   — full-chunk/t1/arms+hands MAE + the literal first action vector for all five
   priority policies. Match ~1e-3 ⇒ serving faithful. Datasets: `human_dp3_dual`,
   `human_dp3_transplant`, `human_dp3c_dual`, `human_dp3_eefframe`, `human_dp3_full_eefframe`.
2. Depth pre-flight (0818 §0a) + LUT blue-basket check (0818 §0b, needed for dp3c).
3. Live cloud QC: stream-1 range med ≈1.3–1.9 m; stream-2 diameter ≤3.0 m; eef
   projection on gripper; dp3c grey-ball check.
4. Static obs, no motion → ≥10 Hz, sane targets.
5. Rollouts: full-task starts, 5 per policy, log furthest stage
   (approach/reach/contact/grasp/lift/place) — grasp-success is THE metric this
   round (the failure mode these designs target).

## 4. Session comparisons worth protecting

- **#2 vs #3 same starts** (embodiment transplant — offline-blind, hardware-only).
- **#1 vs #2 same starts** (does colour survive the domain shift live).
- Any winner vs **0818's dp3_hglass** result (single→dual upgrade on hardware).
- Actions/timing contracts unchanged from 0818 (32×49 @10 Hz, base cumsum, ~1/6 speed).

*Compiled 2026-08-23. All ep1999 ckpts verified on disk; dry-run refs generated by
job 3711901 (serving-path smoke for the new dual/6-D contracts included).*
