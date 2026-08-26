# Point-cloud / depth policy — deployment & sim-eval guide

Covers the **DP3** point-cloud policies and the **Adapt3R** depth policy. For the RGB
policies (v2/v3/v4) see `teleop_deployment_note.md` — the obs contract is different.

---

## 0. Checkpoints and datasets

All paths relative to repo root `/coc/flash7/czhang883/Documents/EgoVerse/`.

| policy | checkpoint | N | dataset it was trained on |
|---|---|---|---|
| **DP3 1024 — RECOMMENDED (latest)** | `logs/RBY1_dp3_pcd1024/dp3_pcd1024_glass_2k/checkpoints/epoch_epoch=1999.ckpt` | 1024 | `datasets/rby1_teleop_pcd1024_glass` |
| DP3 1024 (final) | `.../epoch_epoch=1999.ckpt` | 1024 | same |
| DP3 2048 | `logs/RBY1_dp3_pcd/dp3_pcd2048_glass_2k/checkpoints/epoch_epoch=299.ckpt` | 2048 | `datasets/rby1_teleop_pcd2048_glass` |
| Adapt3R (gray+depth) | `logs/RBY1_adapt3r_slamrect/adapt3r_slamrect_2k/checkpoints/epoch_epoch=1999.ckpt` | — | `datasets/rby1_teleop_slamrect_rgbd` |

**Which DP3 checkpoint — updated 2026-08-02, the 1024 run changed the answer.**

| run | best val | mean val (ep>1500) | clean fit | cloud-reliance |
|---|---|---|---|---|
| DP3 2048 | 0.1112 (ep299) | 0.1184 | 0.0072 | ×25.5 |
| **DP3 1024** | **0.1069** (ep1299) | **0.1150** | 0.0084 | ×17.4 |

**1024 beats 2048 at every stage** and fits training slightly *looser* — the signature of
better regularisation, not more capacity. This matches what the input visualiser showed:
**>50 % of the 2048-point budget lands beyond 1.6 m on background clutter**, which is
memorisable but doesn't transfer. Halving the budget forced FPS to spread points more
usefully.

→ **Deploy `dp3_pcd1024 @ep1999`** (latest-ckpt rule, 2026-08-05: a bit of IL overfit is
fine; @1999 also has the strongest cloud-reliance ×17.4). Best-val reference @1299 =
0.1069 if ever needed; `dp3_pcd2048 @ep1999` for the 2048 contract.

⚠ **N is baked into the checkpoint.** The encoder asserts on point count — a 1024
checkpoint fed 2048 points raises immediately (it will not silently mis-predict). Match
the checkpoint to the cloud size and to the dataset in the same row.

Datasets are LeRobot format; `meta/info.json` carries the feature spec. Sizes: pcd1024
≈ 95 MB, pcd2048 ≈ 190 MB, slamrect_rgbd ≈ 850 MB.

**Tight-crop experiment (run 2026-08-03/04): NEGATIVE.** 0.25–1.5 m range crop lost to
these glass builds at both budgets (tight1024 best 0.1107 vs 0.1069; tight2048 0.1161 vs
0.1112). The recommendation above stands. To deploy a tight policy anyway (hardware A/B),
see `pcd_tight_deployment_guide.md` — the cloud recipe differs (range crop, deterministic
stride) and silently mixing the two contracts would corrupt the input.

---

## 1. Observation contract — DP3

```python
obs = {
    "front_pcd_1":      np.ndarray (1024, 3) float32,   # 1024 for the recommended ckpt; 2048 for the 2048 ckpt
    "robot0_joint_pos": np.ndarray (22,) float32,
    "hand_left_qpos":   np.ndarray (12,) float32,
    "hand_right_qpos":  np.ndarray (12,) float32,
}
```

### The cloud — every one of these matters

| property | required value | why |
|---|---|---|
| shape | exactly `(N, 3)` — **N = 1024 for the recommended ckpt**, 2048 for the 2048 ckpt | encoder asserts on N; wrong N raises immediately |
| dtype | `float32`, **NOT uint8** | the serving code distinguishes cloud-vs-image by dtype+shape |
| units | **metres**, metric | encoder consumes raw xyz; no normalization is applied |
| frame | **GLASS / Aria device frame** | `P_device = T_device_rect @ P_rect` |
| crop | `0.25 m < range < 2.0 m` applied BEFORE sampling | matches training distribution |
| sampling | farthest-point sampling to N | uniform coverage; random subsample ≠ FPS |
| order | irrelevant | encoder is permutation-invariant (verified by unit test) |

**Exact training-time recipe to reproduce (this IS the contract):**
```
depth (512x512, metres, rect frame)
  -> mask 0.25 < Z < 2.0
  -> back-project:  X=(u-cx)*Z/fx,  Y=(v-cy)*Z/fy      with K: fx=fy=307.33668, cx=cy=256.0
  -> random subsample to 16384 (only if more points survive)
  -> FPS to N            # N = 1024 (recommended) or 2048
  -> rotate into glass frame:  P_glass = P_rect @ R.T   where R = T_device_rect[:3,:3]
```
`T_device_rect` is stored in the depth-store attrs (pure rotation, zero translation).

**Do NOT** send the cloud as a `(N,3)` uint8 array or an image-like tensor. The serving
path routes on `ndim==2 and shape[-1]==3 and dtype != uint8`; a uint8 cloud would be
treated as an image, BGR-flipped (xyz → zyx) and divided by 255.

### Proprio (identical to the RGB policies)
- `robot0_joint_pos`: **22-D no-wheel**, order `[torso 6, r_arm 7, l_arm 7, head 2]`.
  Sending the raw 26-D vector fails with `size of tensor a (26) must match b (22)`.
- Hands: 12 each.
- Send **real** proprio. Zeroing it is a valid debug probe (reliance ≈ ×1.0).

---

## 2. Observation contract — Adapt3R (depth)

```python
obs = {
    "front_img_1": np.ndarray (224,224,3) uint8,   # rectified LEFT SLAM image, grayscale
                                                   # replicated to 3 channels, sent BGR
    "aria_depth":  np.ndarray (224,224)  float32,  # METRES, pixel-aligned to the image
    "robot0_joint_pos": (22,), "hand_left_qpos": (12,), "hand_right_qpos": (12,),
}
```
Depth and image must be **pixel-identical rays** — in training they came from the same
rectified stereo frame by construction (`image_rect` + `depth` in the depth store, same K).
`0` = invalid depth. Resize with **NEAREST** only (bilinear invents flying pixels at edges).

---

## 3. Action contract (all policies, unchanged)

`(1, 32, 49)` un-normalized @ 10 Hz:
`base[0:3]` **per-step deltas** → integrate by plain cumsum in the frame-0 heading;
`torso[3:9]`, `head[9:11]`, `l_arm[11:18]`, `r_arm[18:25]`, `l_hand[25:37]`,
`r_hand[37:49]` = absolute joint targets.

---

## 4. Serving

```bash
cd /coc/flash7/czhang883/Documents/EgoVerse && source emimic/bin/activate
python egomimic/scripts/serve_policy.py \
  --checkpoint logs/RBY1_dp3_pcd1024/dp3_pcd1024_glass_2k/checkpoints/epoch_epoch=1999.ckpt \
  --port 8000
```
On connect, metadata must read `embodiment=rby1, action_dim=49, action_horizon=32`.

**Prereqs already fixed in-repo, but verify if serving from another host:**
`msgpack_numpy` and `websockets` must be installed (they were missing from the venv).
Adapt3R checkpoints additionally need the `dinov2` module importable — `serve_policy.py`
now catches `ModuleNotFoundError: dinov2`, warms up `torch.hub.load()`, and retries.

---

## 5. SIM EVAL WITH GT OBS — read this before trusting the result

Sim eval is a **good idea, for a specific purpose**: it is the cheapest way to validate the
*entire deployment stack* — obs construction, frame conventions, action decoding, base
integration — with zero hardware risk. This project has produced a long list of
silent-contract bugs (depth dropped at conversion, xyz flipped by the image path, 26-D vs
22-D proprio, wrong-device calibration), and **every one of them would be caught by a sim
dry-run**. Do it.

**But be careful reading the performance number**, because GT sim obs differ from what
these policies were trained on:

1. **Depth realism gap.** Training clouds came from Fast-FoundationStereo on real stereo:
   they carry stereo noise, edge artifacts, and holes. GT sim depth is perfect. A policy
   trained on noisy depth fed clean depth is **out of distribution** — it may do better
   than reality (flattering) or worse (if it learned to exploit noise structure). Either
   way it is not a real-world estimate.
2. **Scene match.** The policies memorised this specific workspace — grey table, blue
   basket, penguin, that room. If the sim scene differs geometrically, the cloud
   distribution differs and the eval is meaningless. The table height/size and object
   shapes must match.
3. **Cloud construction must be identical.** Same crop, same FPS, same glass frame, same
   metres. If sim gives you a full-scene cloud and you crop differently, you have changed
   the input distribution.
4. **DP3 is geometry-only** — sim textures/colours are irrelevant to it, which actually
   makes DP3 the *most* sim-transferable of our policies. The RGB policies would need
   photorealistic renders matching the rectified camera model to be evaluated fairly.

**Suggested protocol:**
- Step 1: replay a *recorded real* episode through the serving path and confirm predicted
  actions track the logged actions (pure plumbing check, no sim needed). I can script this.
- Step 2: sim rollout with GT obs, built through the exact §1 recipe.
- Step 3: sim rollout with **noise injected into the depth** before cloud construction
  (~1–2 cm Gaussian + random dropout) to approximate the real sensor. If performance
  survives step 3, it is far more likely to hold on hardware.

---

## 6. Live-depth requirement (the real deployment blocker)

On hardware, DP3/Adapt3R need **stereo depth at ~10 Hz on the robot**. Training depth was
computed offline in batch on a SLURM cluster via Fast-FoundationStereo — there is **no
real-time depth code in this repo**. Sim eval bypasses this entirely (GT depth is free),
which is exactly why your plan is a sensible next step: it lets you evaluate the policy
before paying for the perception subsystem.
