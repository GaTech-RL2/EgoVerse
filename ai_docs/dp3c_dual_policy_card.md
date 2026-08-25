# Policy card — dp3c_dual (dual-stream coloured point clouds)

The current best 3D policy on the human corpus. Every number below is measured
(jobs cited); every transform is closed-form and was gated before training.

## 0. Identity

| | |
|---|---|
| run / ckpt | `RBY1_dp3c_dual/dp3c_dual_2k` → `checkpoints/epoch_epoch=1999.ckpt` (deploy) |
| configs | `wb_dp3c_dual.yaml` (experiment) + `rby1_dp3c_dual.yaml` (model), branch `rby1_aria_policy` |
| dataset | `datasets/human_dp3c_dual` (2.2 G; build job 3707038) |
| training data | **135 HUMAN demos only** (fullpp corpus, 53,664 rows). No robot demonstrations. Robot-derived inputs are calibration constants only (mount `M`, glass↔rect rotation) |
| val | 20% episode holdout (27 eps, seed 42) — the family-shared gate |

## 1. Observation (what the network sees, per timestep)

| key | shape | content |
|---|---|---|
| `front_pcd_1` | (1024, **6**) f32 | GLOBAL stream: xyzrgb, **glass frame**, crop = Z∈(0.25, 2.0) m in the estimated-robot-rect frame |
| `front_pcd_2` | (1024, **6**) f32 | LOCAL stream: xyzrgb, **right-eef coords**, crop = 1.5 m ball around right eef (fallback 2.5 m; never fired in 53,664 rows) |
| `eef_pose_glass` | (9,) f32 | right-eef [pos(3), rot6d(6)] in glass frame |
| `robot0_joint_pos` | (22,) | no-wheel joints [torso 6, r_arm 7, l_arm 7, head 2] |
| `hand_left/right_qpos` | (12,)+(12,) | XHand joints |

Colour semantics: **per-point RGB in [0,1]** (albedo-like, viewpoint-independent —
NOT image features). Two colour edits, both mandatory at deploy:
- colour-invalid pixels (LUT/image border) → grey 0.5
- **embodiment neutralization**: rgb → 0.5 for every point within **0.30 m of EITHER
  eef** — the human-hand vs robot-gripper appearance gap deleted; geometry kept.

## 2. Construction pipeline (training side; deploy mirror in §6)

```
per row r of each human demo:
  lift 224² row-aligned depth+colour (human rect frame, K: f=308.052·224/512, c=98)
  → world:  T_fit(demo) ∘ VIO(371-3 @ row ts) ∘ T_devrect_human      [Kabsch fit vs
            retarget world; holdout 2.7 cm vs on-device handtracking]
  → glass:  inv( Tshift(demo) ∘ SE2(base_pose) ∘ FK(base→link_head_2) ∘ M_0726 )
            [FK certified 0.59 mm against stored retarget eef; per-demo origin
            shift solved+asserted <5 mm]
  → colour: attach row image RGB per pixel; grey-fill invalid; neutralize ≤0.30 m of
            either eef (left eef FK gate <20 mm)
  → stream1: rect-Z slab crop → ≤16384 random → FPS 1024 (colour carried by index)
  → stream2: eef-ball crop → ≤16384 → FPS 1024 → re-express by inv(T_glass_eefR)
  → eef_pose_glass = [t, R[:,0], R[:,1]] of T_glass_eefR
```
Build gates that all passed: FK cert both arms, per-demo world fit, **blue-basket
colour presence 135/135 episodes** (automated channel-order check), neutralization
radius assert, rot orthonormality 2.4e-07.

## 3. Architecture & parameters (measured on this ckpt, job 3716871)

Two independent DP3 PointNet encoders (per-point MLP 6→64→128→256, LayerNorm,
max-pool → 1 token each) → per-stream cross-attn stems → shared HPT trunk →
flow-matching head. `point_dropout=0` on both encoders (would decorrelate colour
from geometry); xyz-only pose jitter 5°/2 cm + point noise 1.5 cm kept.

| module | trainable |
|---|---|
| encoder global + local (in_dim=6) | 0.11 M + 0.11 M |
| stems (2 cloud + 4 proprio) | 1.32 + 0.81 M |
| trunk (16 blocks, 256-d) | 12.64 M |
| flow head (CrossTransformer, 32×49) | 6.52 M |
| **total** | **21.54 M** (nothing frozen) |

Proprio leakage treatment (human corpus ⇒ proprio ≈ retarget output): dropout
p=0.9 + noise σ=0.03 + clamp 3.0 on ALL four proprio stems, incl. `eef_pose_glass`.

## 4. Training

AdamW 1e-4 / wd 1e-4, cosine, batch 32, 2000 epochs (14.9 h on one A40 incl. a
requeue), quantile norm computed at start, val every 50 ep. Identical recipe to
every family member — all comparisons isolate the observation design.

## 5. Results (family gate: same 27 episodes)

```
curve: 0.1339@49 → 0.0927@149 → BEST 0.0793@349 → 0.0814@699 → plateau ~0.082 → final 0.0827
manip-phase: best 0.0882@399, final 0.0942
```

| vs | them | delta |
|---|---|---|
| dp3_dual (same, colourless) | 0.0804 / manip 0.0876 | **colour: −0.0011 overall**, manip ≈ tie |
| dp3_transplant | 0.0806 | −0.0013 (offline; transplant's edge is offline-invisible) |
| best single-stream (full_eefframe) | 0.0809 | dual+colour: −0.0016 |
| a3r_eef (colour via frozen DINO) | 0.0840 | per-point colour ≫ image-feature colour |
| h_rect (RGB image) | 0.0651 | RGB still leads offline; it fails on hardware |

Serving-path dry-run reference (final ckpt, real serving, job 3711903):
**full-chunk MAE 0.0051 | t1 0.0058 | arms+hands 0.0059** → `assets_rect_lut/dryref_0823.txt`.

## 6. Deployment contract (robot side)

All transforms exact from robot FK — nothing estimated live:
```
T_base_glass = FK(base→link_head_2) @ M_0726        # verify glasses seating = 0726 solve!
T_glass_eefR/L = inv(T_base_glass) @ FK(base→eef)
image  = LUT remap(raw 640² BGR) → RGB/255          # assets_rect_lut/robot_rect224_lut.npz
depth  = 512² rect → NEAREST 224
cloud  = lift 224 grid → glass → [grey-fill; neutralize ≤0.30 m both eefs]
stream1 = slab crop → FPS 1024 (+rgb) ; stream2 = ball → FPS 1024 → eef coords (+rgb)
```
Serve: `serve_policy.py --checkpoint <ckpt> --port 8010`. Action out (1,32,49)
@10 Hz — base deltas cumsum, rest absolute; executes at ~1/6 human speed (60 Hz
rows labeled 10 — expected). Full recipe + self-checks: `hw_session_0823_guide.md` §2b.
Robot-side self-checks: basket points blue-dominant; points near eefs exactly 0.5;
stream-2 diameter ≤3.0 m; replay matches dryref to ~1e-3.

## 7. Known limitations / open questions

- Offline val contains human arms & human colour — the domain-shift robustness that
  motivated the neutralization is only partially measurable offline; hardware
  (0823 session, priority 1) is the real test.
- Colour is albedo-as-seen-from-human-views: lighting/shading differences to robot
  views survive neutralization (user's standing concern — empirically small offline,
  unknown live).
- Peak checkpoint is ep349 (0.0793); deploy rule uses ep1999 (0.0827). Drift is
  mild (+0.0034) but present.
- Mount seating: dataset built with the 0726 `M`; re-seated glasses ⇒ re-solve
  before rollout (`solve_head_mount.py`).
