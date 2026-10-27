# Training-data sanity check — handoff (2026-09-14)

Goal: visually audit every demo the current policies were trained on, to find data
defects that could explain weaker-than-expected policy performance. All paths are
absolute and on `sky2.cc.gatech.edu`.

---

## 1. THE dataset the policies trained on

```
/coc/flash7/czhang883/Documents/EgoVerse/datasets/dp3c_mix186          # 3.1 GB, LeRobot v2
```
186 episodes = **150 train + 36 val**, 75,386 rows total (train 61,926 / val 13,460).

| corpus | train episodes | what it is |
|---|---|---|
| `old` | 99 | original 135-demo human pick&place corpus (July), sampled with Random(42) |
| `chendong` | 30 | Chendong_toy (2026-09-07), cam height ~171 cm |
| `chuye` | 21 | Chuye_diverse_toy (2026-08-28), cam height ~160 cm |

The 36 val episodes are ALL from the `old` corpus and were never trained on by any
policy generation (they contain the entire historical 27-episode leaderboard val).

**Variants of the same 186 episodes** (identical split, different observations):
```
datasets/dp3c_mix186            # baseline: 1024-pt clouds, 0.30 m neutralization, NO trim
datasets/dp3c_mix186_2048       # 2048-pt clouds
datasets/dp3c_mix186_trim       # + partial raise-trim (see §4 — known too weak)
datasets/dp3c_mix186_trim_n15   # trim + 0.15 m neutralization
datasets/dp3c_mix186_trim_n45   # trim + 0.45 m neutralization
datasets/dp3c_mix186_v2         # BUILDING NOW: proper trim + 1.0 m eef balls + left-eef stream + SG actions
```
Which policy used which: v1/v2/v4 → `dp3c_mix186`; v3 → `_2048`; v5 → `_trim`;
v6 → `_trim_n15`; v7 → `_trim_n45`; v8/v9 → `_v2`.

### Per-episode provenance (the key file for this audit)
```
/coc/flash7/czhang883/Documents/EgoVerse/ai_docs/assets_rect_lut/mix186_manifest.csv
```
One row per episode slot: `slot, split, corpus, src_ep, demo, recording, rows, vrs`
— so any suspicious episode traces straight back to its source VRS recording.

### Component datasets (pre-mix, if you want to inspect a corpus in isolation)
```
datasets/human_dp3c_dual      # old 135, 1024-pt coloured dual streams
datasets/toy_chuye_dp3c       # 21 demos
datasets/toy_chendong_dp3c    # 30 demos
datasets/human_fullpp_rgbd    # old 135 WITH rectified RGB + depth (row-parallel)
datasets/human_gazerect       # old 135 full-res RGB + eye gaze (16 GB)
```

### Observation schema in the mix
| key | shape | meaning |
|---|---|---|
| `obs.aria_pcdc` | 6144 = 1024×6 | GLOBAL cloud, xyz+rgb, **glass/device frame** |
| `obs.aria_pcdc_local` | 6144 | LOCAL cloud, xyz+rgb, **right-eef frame** |
| `obs.eef_pose_glass` | 9 | right-eef pos + rot6d in glass frame |
| `obs.robot0_joint_pos_no_wheel` | 22 | body joints |
| `obs.hand_left_qpos` / `_right_` | 12 each | hand joints |
| `actions.joint_base_torso_head_arm_hand` | 49 | **the training target** (see trap below) |

⚠ `actions.joint` also exists and is ALSO 49-D but is **not** the target — scoring
against it makes a working policy look ~3.7× worse.

---

## 2. Cropping criteria as actually implemented (verified, 0 violations in 192 frames)

- **GLOBAL**: kept if `0.25 < z_rect < 2.0 m`, where `z_rect = (P_glass @ R_DEV_RECT)[:,2]`
  (depth along the rect optical axis). **Depth-only — there is NO x/y crop.** Lateral
  extent is bounded solely by the 79.6° camera frustum: 31% of points sit >1.0 m
  off-axis, 14% >1.25 m. Possible defect source worth eyeballing.
- **LOCAL**: kept if `‖p − p_eefR‖ < 1.5 m` (v2 build: **1.0 m**), then rigidly
  re-expressed in the right-eef frame. Ball is always full; the <64-point fallback
  never fires.
- **NEUTRALIZATION** (colour only, geometry untouched): rgb → 0.5 within **0.30 m of
  either eef**, both streams. Verified by radius profile: grey fraction is 1.00 inside
  the radius and drops immediately outside.
- Both streams are farthest-point-sampled to exactly 1024 points.

---

## 3. Tools to visualize (all exist, all headless-capable)

**Interactive dual-stream viewer** (what the policy actually eats, orbit + scrub):
```
template : <session scratchpad>/final_inputs_viewer.html
data     : /coc/flash7/czhang883/tmp/final_inputs_viewer.json   (2.0 MB, one episode)
published: https://claude.ai/code/artifact/ff7a68f2-7d55-46fe-8436-4a523411fdfd
```
Regenerate for any episode by re-running the exporter with a different slot.

**Open3D sphere-style renders** (dark bg, true RGB, solid matte spheres):
```
/coc/flash7/czhang883/tmp/o3d_real_clouds.py      # env: DS, EP, STRIDE  -> mp4 + png
/coc/flash7/czhang883/tmp/o3d_real_clouds.sbatch  # GPU node (EGL); ~2 min/episode
```
Rendering traps already solved in that file: compute normals ONCE on an expanded
layout (else every sphere renders black); park empty clouds at the anchor (not far
away, which blows the auto-fit camera); `setup_camera(fov, center, eye, up)`.

**Chunk-quality / label-trust viewer** (predicted eef trajectory vs labels on RGB+cloud):
```
/coc/flash7/czhang883/tmp/xembody_chunkviz.sbatch   # env: CKPT, TAG, EP, NEUT
```

**Action-curve plots** (smoothness per 49-D block):
```
/coc/flash7/czhang883/tmp/audit_smooth.sbatch  -> action_curves.png, action_zoom.png
```

---

## 4. Known defects — check these FIRST, they are the leading suspects

1. **Terminal hand-raise (stop-recording gesture) is still in the toy demos.**
   The EgoPoser pipeline's `trim_hdf5.py` only removes *frozen-tracking* tails; the
   raise has valid tracking and passes through. Measured: 100% of the 51 toy episodes
   ended with the wrist <0.40 m from the glasses (median 0.223 m) before trimming.
   - `dp3c_mix186` (used by v1–v4): **NO trim at all** — the raise is fully present.
   - `dp3c_mix186_trim` (v5–v7): partial trim, still 43% (chuye) / 20% (chendong) of
     episodes ending inside 0.40 m, vs the old corpus's 12%.
   - `dp3c_mix186_v2` (v8/v9): proper descent-ramp trim → 14% / 0%.
   **Visual check**: play the last ~2 s of any toy episode; a hand rising to the face
   is the artifact. Old-corpus episodes should NOT show it.
2. **Domain gap between corpora.** The 99 old + 51 toy demos are different scenes,
   tasks and collectors. Measured fact: adding the toy data did NOT improve old-domain
   val (v1 0.0868 vs old checkpoint 0.0847 on identical episodes). Worth checking
   visually whether toy scenes/objects/lighting look consistent with the old corpus.
3. **Collector body-height mismatch (toy sets).** EgoPoser retargets onto a fixed
   mean-shape avatar (head joint 1.592 m). Chuye ≈160 cm ⇒ near-neutral; Chendong
   ≈171 cm ⇒ ~7% scale mismatch, which can distort retargeted reach. No
   `--subject_scale` normalization has been applied yet.
4. **Peripheral clutter in the global cloud** (no x/y crop, §2) — a third of points
   are >1 m off-axis and spend FPS budget that could go to the workspace.
5. **1/6-speed replay convention**: human rows are 60 Hz labelled 10 Hz, so policies
   replay ~6× slower than real time. Expected, not a defect.
6. **One bad recording**: `unknown_20260719_165754` failed BOTH the flow gate and the
   gaze probe (worst of 135). Recommend excluding it from future training sets.

## 5. Suggested audit order
1. Render the last 3 s of all 51 toy episodes → confirm/deny the raise in `dp3c_mix186`.
2. Spot-render 10 old + 10 toy episodes in the same style → look for scene/scale/colour
   inconsistency between corpora.
3. Check the val-36 episodes specifically (they set every number we quote).
4. Report any episode whose cloud looks empty, truncated, mis-framed, or grey-dominated;
   trace it via `mix186_manifest.csv` back to its VRS and flag the recording.
