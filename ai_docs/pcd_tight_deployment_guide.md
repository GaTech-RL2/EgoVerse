# TIGHT-crop DP3 policies — deployment guide

Delta guide to `pcd_policy_deployment_guide.md` (the glass-frame DP3 guide). Everything
not mentioned here — action contract, serving, proprio, sim-eval caveats, live-depth
blocker — is **identical** to that document. This one covers only what the tight builds
changed and which checkpoints to serve.

---

## 0. Verdict first (2026-08-04, both runs finished)

Tight crop **lost to the glass-frame builds at both point budgets** on the identical
held-out val:

| run | best val (ep) | tail mean (ep>1680) | glass-frame counterpart |
|---|---|---|---|
| dp3_tight1024 | 0.1107 (599) | 0.1235 | **0.1069** (pcd1024_glass, ep1299) |
| dp3_tight2048 | 0.1161 (249) | 0.1316 | 0.1112 (pcd2048_glass, ep299) |

The "background points are wasted budget" hypothesis is closed: removing them did not
help — the far points were carrying context, not just noise. **For a single best DP3
deployment, use `dp3_pcd1024_glass @ep1299` from the main guide.** Deploy a tight policy
only to A/B this finding on hardware (offline val has been wrong about hardware before —
see the hd_wam3 vs hd_resnet flip).

## 1. Checkpoints

| policy | checkpoint (repo-root relative) | val there |
|---|---|---|
| **tight 1024 — deploy (latest)** | `logs/RBY1_dp3_tight1024/dp3_tight1024_2k/checkpoints/last.ckpt` | ~0.122 tail |
| tight 2048 — deploy (latest) | `logs/RBY1_dp3_tight/dp3_tight2048_2k/checkpoints/last.ckpt` | ~0.132 tail |
| best-val references | `...tight1024...epoch_epoch=599.ckpt` (0.1107) / `...tight2048...epoch_epoch=299.ckpt` (0.1223) | fallback only |
| datasets | `datasets/rby1_teleop_pcd1024_tight` / `datasets/rby1_teleop_pcd2048_tight` | |

Checkpoint rule (user decision 2026-08-05): deploy the LATEST checkpoint — mild IL
overfit is acceptable. (tight2048's best measured val 0.1161@249 was never saved;
the xx99 grid + best-val rows above are reference/fallback only.)

N is baked into the checkpoint exactly as in the main guide — a 1024 ckpt fed 2048
points raises immediately.

## 2. The cloud — what changed vs the glass recipe

Two lines of the main guide's §1 recipe are different, plus one fallback:

| step | glass builds | **tight builds (this guide)** |
|---|---|---|
| crop | Z-slab: `0.25 < Z < 2.0` | **Euclidean RANGE: `0.25 < sqrt(X²+Y²+Z²) < 1.5`** |
| pre-FPS subsample | random 16384 | **deterministic stride to ≤16384** |
| degenerate frame | — | if crop leaves **< 64 points**: re-crop at `0 < range < 3.0` |

Everything else is unchanged: same K (fx=fy=307.33668, cx=cy=256.0 at 512), same
back-projection, same FPS to N, same rect→glass rotation, metres, float32.

Why these matter at deploy time:
- **Range, not Z.** A Z-slab keeps wide off-axis floor at large true distance (at
  Z=1.5 m the image corner sits at range 2.31 m). Cropping the live cloud on Z while the
  training cloud was cropped on range shifts the input distribution at exactly the
  boundary the policy is most sensitive to.
- **Deterministic stride, not random.** The training builds switched because random
  subsampling made the cloud flicker: chamfer between two resamples of the *same* depth
  frame was 22 mm ≈ 81% of the real frame-to-frame change. A live random subsample would
  re-introduce input noise the policy never saw in training.
- **The <64-point fallback** fires when the operator looks away from the workspace
  (nothing inside 1.5 m). Training data contains such frames (built with the same
  fallback), so replicate it rather than sending a padded near-empty cloud.

**Exact deploy-time recipe (mirrors the build, `tmp/build_pcd_tight.sbatch`):**

```python
# depth: (512,512) float32 metres, rect frame;  K: fx=fy=307.33668, cx=cy=256.0
X = (u - cx) * Z / fx ;  Y = (v - cy) * Z / fy            # back-project full grid
rng = sqrt(X*X + Y*Y + Z*Z)
m = (Z > 0.05) & (rng > 0.25) & (rng < 1.5)               # RANGE crop
if m.sum() < 64:                                          # degenerate: widen
    m = (Z > 0.05) & (rng < 3.0)
P = stack([X[m], Y[m], Z[m]], axis=1).astype(float32)     # raster order — keep it
if len(P) > 16384:                                        # DETERMINISTIC stride
    step = len(P) // 16384 + 1
    P = P[::step][:16384]
P = fps(P, N)                                             # N = 1024 or 2048, match ckpt
if len(P) < N:                                            # pad by resampling
    P = concat([P, P[randint(0, len(P), N - len(P))]])
P_glass = P @ R.T                                         # R = T_device_rect[:3,:3]
```

`fps` = the same farthest-point sampling as training
(`egomimic.models.adapt3r_3d_encoder.fps_pytorch`). The stride step assumes `P` is in
raster (row-major pixel) order — do not shuffle before striding.

## 3. Obs / action contract — unchanged

```python
obs = {
    "front_pcd_1":      (N, 3) float32, metres, GLASS frame,   # N matches the ckpt
    "robot0_joint_pos": (22,) float32,   # no-wheel order [torso 6, r_arm 7, l_arm 7, head 2]
    "hand_left_qpos":   (12,) float32,
    "hand_right_qpos":  (12,) float32,
}
```

Same dtype-routing warning as the main guide: a uint8 cloud gets treated as an image
(BGR-flip + /255). Action output `(1, 32, 49)` @10 Hz, base deltas integrated by plain
cumsum in the frame-0 heading — see main guide §3.

## 4. Serving

```bash
cd /coc/flash7/czhang883/Documents/EgoVerse && source emimic/bin/activate
python egomimic/scripts/serve_policy.py \
  --checkpoint logs/RBY1_dp3_tight1024/dp3_tight1024_2k/checkpoints/last.ckpt \
  --port 8000
```

Connect metadata must read `embodiment=rby1, action_dim=49, action_horizon=32`.
Offline replay smoke test (recommended before hardware, main guide §5 step 1):

```bash
python egomimic/scripts/test_serve_policy_client.py \
  --episode-idx 0 --max-steps 30 \
  --dataset-folder datasets/rby1_teleop_pcd1024_tight --trajectory
```

This replays *training-format* clouds through the real serving path — it validates
plumbing, not the live crop code. To validate your live cloud construction, run your
deploy-side recipe on a stored depth frame and check chamfer against the corresponding
`obs.aria_pcd` row (< ~5 mm expected; the pipeline is deterministic end-to-end apart
from FPS's data-independent start point).
