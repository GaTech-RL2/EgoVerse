# PROJECT BRAIN — complete knowledge handoff (2026-08-26)

Everything currently known about the RBY1 human→robot policy project, written to be
picked up cold by a future agent or collaborator. Deep-dive docs are referenced, not
duplicated; findings and state that live nowhere else are written out in full.

---

## 0. Mission & current phase

Train whole-body (49-D) RBY1 policies from **human demonstrations only** and make
them work zero-shot on the real robot. The through-line discovery of the project:
**the human→robot observation gap, not model capacity or data volume, is the
bottleneck — and it is closable with closed-form geometric transforms** because the
SEW retargeting gives us, for every human frame, where the robot's body *would be*.

Current phase: the "frame-transform campaign" (13 human-corpus policy variants) is
COMPLETE offline; the 0823 hardware session (dual/colour/transplant fleet) is the
next real-world readout. Three external handoffs are in flight (§7).

## 1. Data corpus knowledge

Canonical docs: `ai_docs/rgbd_data_handoff.md` (formats/frames/semantics),
`ai_docs/DATA_INVENTORY.md` (provenance), `ai_docs/DATA_TRANSFER_0825.md` +
`/coc/flash7/czhang883/handoff_all/` (transfer bundles + symlink farm).

Facts that repeatedly matter:
- **Human corpus**: 135 demos / 53,664 rows (glove/bare × nav/pp), colour-rect RGB
  + Fast-FS depth, **rows are 60 Hz labeled 10 Hz** → policies replay at ~1/6 human
  speed (expected at deploy, not a bug). Unique frames ≈ 9k RGB / 6.8k depth.
- **Teleop corpus**: 63 demos / 7,003 rows (near-table manip only), true 10 Hz.
- **Traps that cost debugging cycles**: (1) `actions.joint` ≠
  `actions.joint_base_torso_head_arm_hand` — both 49-D, only the latter is the
  policy target; scoring the wrong one looks like a 3.7× policy failure.
  (2) LeRobot episode order = **string-sorted** demo names (demo_1, demo_10, …).
  (3) robot-recorder images are BGR, human RGB; frozen pretrained encoders must get
  RGB. (4) depth 0=invalid, NEAREST resize only. (5) human proprio ≈ retarget output
  ≈ action label → leakage; hence dropout 0.9 on all proprio stems.
- Raw 512² depth: teleop `depth_store/` (uint16 mm + calib attrs), human
  `fastfs_fpp_depth_npy/` (float32 m, device-ts filenames). Both on scratch —
  the one non-reconstructible-without-GPU-hours asset if scratch is cleaned.

## 2. Transforms & calibration (the closed-form toolbox)

Single machine-readable source: `/coc/flash7/czhang883/handoff_calib/aria_transforms.json`.

| transform | value/source | validation |
|---|---|---|
| rect intrinsics | robot f=307.336684, human f=308.052437, c=256@512, pinhole | device constants, spread 0 |
| glass↔rect (robot) | pure rotation `R_DEV_RECT` (in json + depth-store attrs) | DP3-glass hardware-validated |
| robot RGB model | `robot_rgb_solved.npz` (rect→640 RGB, pinhole+k1k2) | 0.43 px held-out; independently confirmed by factory-calib chain to 0.3% |
| colour-rect LUT | `assets_rect_lut/robot_rect224_lut.npz` (remap 640 BGR→224 rect) | 1.44 px vs depth-exact pipeline |
| head mount M | `head_mount.json` per capture day (0726 in use) = T_head2_device | arm-hit holdout; **re-solve if glasses re-seated** |
| human world chain | per-demo Kabsch fit of VIO(371-3) odom → retarget world | holdout **2.7 cm** vs on-device handtracking |
| robot-config FK | retarget's own MuJoCo model (model_v1.3_xhand_act.xml); needs TORSO+HEAD+ARM joints set; **per-demo XY origin shift** solved from stored eef | certified **0.59 mm** vs stored eef |
| eef definition | body `right_eef`: palm-centre point, 10.2 cm distal of wrist joint, 6.5 cm from hand root; ≈ human wrist (2–3 cm) | model + handtracking |
| live raw RGB frame | **640×640 as streamed** (lawrence_custom profile); robot native = 2016×1512 (crop 1512²@x=252 →640) | publisher code; NOT 2560×1920 (that's the human device) |

Gate philosophy that made this work: every transform gets a quantitative gate +
independent ground truth BEFORE any training consumes it; asserts in build jobs,
not post-hoc checks. Bugs caught this way: episode string-sort, per-demo origin
shift, missing arm joints in FK (155 mm → 0.59 mm), eefball↔episode misalignment.

## 3. The policy fleet — final numbers (shared 27-episode val, seed 42)

Spec table: see the variant-spec message/`dp3c_dual_policy_card.md`; observation
figures: `tmp/all_variants_obs.png`, `tmp/dp3c_streams.png`, `tmp/neutral_zoom.png`,
video `tmp/dp3c_dual_streams.mp4`.

| policy | obs | best / final | note |
|---|---|---|---|
| h_rect | RGB image | **0.0651** / 0.0656 | offline king; weak in real world |
| **dp3c_dual** | 2×1024 xyzrgb + eef pose | **0.0793** / 0.0827 | offline 3D champion |
| dp3_dual | 2×1024 xyz + eef pose | 0.0804 / 0.0860 | best manip 0.0876 |
| dp3_transplant | dual, human arms→robot arms | 0.0806 / 0.0849 | offline-tied BY DESIGN (val has human arms) |
| dp3_full_eefframe | 1×1024 xyz, full scene, eef coords | 0.0809 / 0.0946 | best single-stream |
| dp3_dual_noprop | dual, no eef pose | 0.0816 | eef-pose worth ~0.001 |
| dp3_dual_pos3 | eef pos-only 3D | 0.0823 | rot6d worth ~0.002 |
| eefframe / hglass / eefball | single-stream frame ablation | 0.0824–0.0828 | frames tie offline |
| a3r_eef | Adapt3R, eef frame | 0.0840 / 0.0855 | beats camera-frame by 0.006 |
| dp3_dual_eefonly | no joint angles | 0.0841 | joints still worth 0.004 |
| a3r_human | Adapt3R, camera frame | 0.0895 / 0.0903 | |
| Teleop refs | v2 0.0833 · v4 0.0859 · dp3 glass 0.1069 · a3r grey 0.1157 / colour 0.1224 | | different corpus |

Hardware so far (user sessions): teleop v2 grasped (validated); human RGB
navigates/reaches but fails task; **dp3_hglass (robot-frame transfer) = best human
policy on hardware** — partial success, misses grasps, weak manip-phase awareness
(→ motivated the dual/colour/transplant round). 0823 fleet not yet rolled out.

## 4. Findings (the knowledge, ranked by importance)

1. **Frame is the lever.** Re-expressing observations in robot-native / eef frames
   closed ⅓ of the 3D-vs-RGB gap (0.090→0.079-0.084) at zero data cost. Camera-frame
   Adapt3R had skipped Adapt3R's actual mechanism (eef re-framing was an unimplemented
   TODO — found in the port's docstring).
2. **Dual local+global fixes manip prediction** (~10% manip-MAE gain). Impact probe
   (96-step ablation, `tmp/dp3c_impact_static.png`): local-stream impact > global at
   100% of steps (no crossover — real; local ball doubles as the phase signal during
   nav); both rise in manip; proprio impacts ≈0.0001–0.0004 (policy is cloud-driven)
   EXCEPT right-hand qpos spiking ~0.002 exactly at grasp instants.
3. **Per-point colour wins offline** (dp3c 0.0793) — colour as albedo into a
   from-scratch PointNet, NOT via frozen image features (the teleop "colour ceiling"
   0.1224 was the encoder, not the colour: proven by the solved-calib colour rerun).
   Embodiment colour-excision: rgb→0.5 within 0.30 m of either eef, train AND
   deploy (hides human hand at train, robot gripper at deploy — same observation
   both sides). Residual gap: forearm colour beyond 0.30 m; capsule grey-out is the
   ready fix. Our design, not from a paper (3D analog of image-space hand masking).
4. **Transplant (arms swapped in-cloud) is offline-invisible** — only hardware or a
   robot-obs probe can score it. Cross-embodiment probe result (pre-dual fleet):
   human policies scored on real robot obs — a3r_eef-style eef framing helped;
   neither RGB nor camera-frame beat episode-mean; a3r_human 0.204 < h_rect 0.221
   (reverse of offline). The dual-fleet robot-obs probe has NOT been run — planned.
5. **DP3 "attention" = max-pool critical points** (exact, one winner per channel);
   Adapt3R pools to ONE token (stem attn trivially uniform), its internal pooling
   attention is diffuse (entropy 0.66–0.78) vs ResNet policy locking onto objects
   (0.58). DP3 critical points sit on table edges/contours, not objects → why
   colourless DP3 misses grasps.
6. **Adapt3R underperformance is unexplained** — handed to Dennis with ranked
   suspects: S1 RGBD path skips ALL image augs (live TODO(aug) in hpt.py); S2 NeRF
   pos-enc fed raw metres (original uses normalized bounds — aliasing beyond ~2 m);
   S3 no scene crop; S4 vision-path capacity 0.1 M vs the ~10 M "capacity law" from
   the HD round; S5/S6 width/pooling. See fork's `ai_docs/ADAPT3R_HANDOFF.md`.
7. **Proprio dropout 0.9 everywhere** (leakage: human proprio ≈ action). Open
   design question: `eef_pose_glass` could justify lower dropout (0.3–0.5) since
   deploy-side it's exact FK — one-run ablation, queued idea, not launched.
8. **Latest-ckpt deploy rule** (user): always ep1999/last, mild IL overfit accepted.
9. DP3 encoder augs: per-cloud pose jitter 5°/2 cm about the CENTROID + 1.5 cm point
   noise; point_dropout 0.1 for xyz, **0.0 for coloured** (would decorrelate
   colour↔geometry); streams augmented independently; no colour aug (known gap).

## 5. Architecture facts (measured)

dp3c_dual: 21.54 M trainable, 0 frozen — 2× DP3PointNet encoders (**separate
weights**, 0.11 M each, in_dim=6), 2 cloud stems + 4 proprio stems, trunk 12.64 M,
flow head 6.52 M. Full card: `ai_docs/dp3c_dual_policy_card.md`.
Encoder restore-safety: `in_dim` via getattr default 3 (pre-patch ckpts unpickle
without the attr — was a live crash, fixed+pushed). Serving routes (N,3) and (N,6)
float clouds; raises if a model's `eef_T` extrinsic is required but missing.

## 6. Deployment state

- Guides: `hw_session_0823_guide.md` (current fleet: recipes for dual/colour/
  transplant incl. the FK chain, eef-ball/coords construction, colour rules),
  `hw_session_0818_guide.md` (constants + h_rect/eefball/a3r_eef),
  `a3r_human_realworld_eval_guide.md` (single-policy, depth-sensitivity-driven),
  `dp3c_dual_policy_card.md` (§6 = its contract). Dry-run refs on FINAL ckpts:
  `assets_rect_lut/dryref_0823.txt` (all five 0.0051–0.0069) + `dryref_new3.txt`.
- Headline hardware experiment queued: **dp3_dual vs dp3_transplant, same starts**
  (the offline-blind arm-swap test) + dp3c vs dual (does colour survive the shift).
- Robot-side mandatory live ops for dp3c: LUT colour (RGB [0,1]), grey-fill invalid,
  **0.30 m grey-out of both eefs every frame** (robot FK exact), 224-grid lift,
  crops+FPS per stream, eef_pose_glass 9-D.
- Standing caveats: mount seating vs 0726 solve; ~1/6-speed replay; depth
  knife-edge (10% scale error worse than no depth — measured on a3r_human).
- Deploy-side epistemics: everything estimated at training (VIO fits, Kabsch,
  origin shifts) is EXACT on the robot via its own FK — the asymmetry that makes
  the whole approach deployable.

## 7. Handoffs in flight

1. **Dennis / Adapt3R investigation**: fork `ZhangChuye/EgoVerse@rby1_encoder_dev`
   — full Adapt3R stack + `ADAPT3R_HANDOFF.md` (file-map/usage manual style, per
   user preference; suspects live in git history + §4.6 above). Earlier drops on the
   same branch: encoder-dev kit, WAM quickstart, ConvNeck/d3conv configs,
   `rgb_policy_structure_d3conv.md`, `rgbd_data_handoff.md`.
2. **Labmate data transfer**: `handoff_all/` symlink farm (tier1 ~30 G essentials /
   tier2 ~35 G) + `handoff_calib/` bundle; rsync -avPL commands in
   `DATA_TRANSFER_0825.md`; timeouts → rsync resume + ServerAlive options.
3. **Hardware agent**: 0823 guide + refs + pushed code (main repo
   `GaTech-RL2/EgoVerse@rby1_aria_policy`, latest commits incl. serving guards and
   restore-safety fix). Their side owns: mount check, live recipes, rollout ladder.

## 8. Conventions & infrastructure

- Training: sbatch on overcap, `--requeue` + afterany "insurance" clone per run;
  smokes via trainer=debug (needs ≥3 h — norm-stat inference over 53k rows eats ~1 h);
  preemption weather is real; DependencyNeverSatisfied after preemption → relaunch
  manually. 2000 ep default; val every 50; ~90–110 ep/h on A40.
- Reused job templates in `/coc/flash7/czhang883/tmp/*.sbatch`: build_dp3c_dual
  (the full VRS+FK+colour build), build_transplant, build_eefframe (rigid
  re-expression — FPS is rigid-invariant), assemble_* (parquet column ops),
  param_census, a3r_depth_probe, attn_new4, xembody_eval, dryref_*, viz/render
  scripts (viz_all_variants, viz_dp3c_streams, render_dp3c_video, zoom_neutral3,
  export_viewer_data + the artifact viewer page in the session scratchpad).
- Val protocol: split_dataset_names(sorted names, Random(42)); human 20% (27 eps);
  gate metric `Valid/rby1_..._mae_avg` + `_manip_` variant (manip = GT-chunk base
  disp < 0.05 m).
- wandb: project sew_policy, personal entity; run id = name_description.
- No login-node compute; no NVS code in any repo (audit every push); memory rules.

## 9. Open questions / next-action queue

1. 0823 hardware session → grasp-rate per policy; dual-vs-transplant verdict;
   colour survival. (The decisive unknowns.)
2. Robot-obs probe for the dual fleet (teleop-side dual observations via real FK)
   — offline preview of #1; build job sketched but not run.
3. eef_pose dropout 0.3–0.5 ablation (one run).
4. Forearm capsule grey-out variant if colour underperforms live (build = transplant
   masks as colour-op; ~1 day).
5. Colour augmentation for dp3c (none currently).
6. Dennis's Adapt3R verdict → fold fix into a3r_eef.
7. Deferred: teleop-as-val for human policies; wrist/1.5m-LUT ideas; scrub of git
   history pre-473f0cb1 (NVS) never done.

*Compiled 2026-08-26 from the full working session. Companion persistent-memory
pointer lives in the agent memory dir; this file is the source of truth.*
