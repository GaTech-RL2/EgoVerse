# PROJECT STATE — full handoff (2026-09-11)

Continuation of `PROJECT_BRAIN_0826.md` (still valid for everything up to the 0823
fleet; read it first). This doc covers everything since: the dp3cmix round, the new
toy data, the retarget-pipeline reproduction, the LUT fix, gaze, flow, monocular
depth, and the visualization tool suite.

---

## 1. dp3cmix round — new data + ablations (the current fleet)

**Data**: 150-demo train mix = 99 sampled old (Random(42)) + 51 new toy demos
(Chuye_diverse_toy 21 @160cm, Chendong_toy 30 @171cm); val = the other 36 old demos.
Seed-42 luck: the new 36-ep val CONTAINS the entire old 27-ep leaderboard val, so
cross-generation comparison is direct. Split is PLACEMENT-controlled (loader always
re-shuffles with seed 42; assembly puts val episodes at the slots the shuffle picks;
`valid_ratio=0.194` → exactly 36). Datasets: `dp3c_mix186`, `_2048`, `_trim`,
`_trim_n15`, `_trim_n45` (75,386 rows; trim variants 73,332).

**Final results (2000 ep, shared-27 serving-path MAE, final ckpts —
`assets_rect_lut/dryref_mix.txt`)**:
v4_h128 first-32 **0.0815** < v2_drop07 **0.0821** < v3_2048 0.0843 <
old dp3c_dual (identical eval) **0.0847** < v1_base 0.0868.
Findings: (1) proprio dropout 0.7 > 0.9, robust ~0.006; (2) horizon-128 training
improves even the first 32 steps (auxiliary-task effect) — best overall; (3) 2048 pts
≈ wash; (4) v1 vs old ckpt: toy data alone does NOT help old-domain val — knob gains,
not data volume, drove this round. Deploy rec: v2 + v4 vs old dp3c_dual same-start A/B.

**In flight (resubmitted 09-11 after REQUEUE_HOLD "launch failure limit" — system
hold, admin-only release; cancel+resubmit with resume is the workaround)**:
- v5 = original recipe on `dp3c_mix186_trim` (~ep1600/2000)
- v6 = trim + NEUT 0.15 (~ep1500), v7 = trim + NEUT 0.45 (~ep1500)
**Trim** = the raise-cut the toy data was missing (terminal hand-raise-to-stop;
wrist-to-glasses <0.40 m from `work/*_aria60.npz`, onset −0.3 s; 37/51 episodes cut
31–73 frames; old corpus already carried it). v1–v4 all trained UNTRIMMED → their
rollouts can show a learned end-of-task hand raise (data artifact, not policy bug).
**NEUT variants** = colour rebuilds from source at 0.15/0.45 m grey-out (smaller radius
is NOT derivable from 0.30 data — colour already destroyed). Live grey-out radius at
deploy must match the ckpt: 0.30 (v1–v5) / 0.15 (v6) / 0.45 (v7).

**Ckpts**: `logs/RBY1_dp3cmix_v{1..7}/mix150_*/checkpoints/last.ckpt`. Serving
contract = dp3c policy card §6; v3 needs 2048-pt FPS/stream; v4 emits 128×49 chunks.
Dry-run examples: `assets_rect_lut/dp3cmix_example_{1024,2048}.npz` (val frame,
never trained on, + GT future actions).

## 2. New data & the reproduced retarget pipeline

- VRS→HDF5 pipeline (EgoPoser + WARP/SEW) reproduced on Skynet: worktree
  `Documents/SEW-aria-pipeline` @93cc496, envs aria2/egoposer/rby1 (miniconda3),
  handoff bundle `mobile_generalist/data/aria_egoverse... (aria_egoposer_warp_handoff/)`.
  Certified vs reference: joints ≤2.4e-5 rad, images/ts bit-exact.
- Toy sets built end-to-end here: `Chuye_diverse_toy_hdf5` (21 demos),
  `Chendong_toy_hdf5` (30), zero failures; + Fast-FS depth
  (`scratch/aria_fs_out/fastfs_toy_depth_npy`), LeRobot (`datasets/toy_ch*`),
  dp3c clouds (`toy_*_dp3c{,_2048,_n15,_n45}`).
- **Height semantics (author-verified)**: the 1.60 z-lift = SMPL mean-shape HEAD-JOINT
  height (1.592 m; stature 1.79). Pipeline feeds device position AS the head joint
  (OFF_HEAD is rotation-only). Per-collector heights → `--subject_scale` INPUT
  normalization (never the z-lift). Convention ambiguity (device-as-head-joint vs
  like-for-like) flips who is "mean-shaped" (Chuye 160 vs Chendong 171) → resolve
  EMPIRICALLY: sweep s ∈ {0.93, 0.98, 1.00} on Chendong, score EgoPoser wrist error +
  TCP quality. Blocked on: `--subject_scale` patch (laptop agent offered) and it
  doesn't block training (toy baseline bias accepted for the A/B).
- Depth floor-fit cross-check of Chuye's ~1.60 m: consistent within method ±10 cm
  (estimator: floor-plane band 1.2–2.0 m below device, VIO gravity, VALID records only;
  medians catch the table at ~0.85 m and bending — use standing plateau).

## 3. Rect-colour LUT thread (deploy-side, from PROJECT_BRAIN addendum era)

Old solved LUT over-blacks 9.45% (polynomial tail extrapolation past 45°; true
sensor-edge limit 50.1°). Fix = regenerate from factory FISHEYE624
(`assets_rect_lut/regen_rect_lut_factory.py`, validated on human device IoU 0.94–0.97)
— WITH the hardware agent, not yet confirmed back. Canonical mask
`assets_rect_lut/rect224_invalid_mask.npz` (arc_strict 4.06% / arc_union 4.18% /
mask 4.90%, covers all 135 recordings 0 px missed). Corpus audit: training data
needs NO repair. dp3c colour impact of the bug: only 2.9–3.7% of points lose colour
(measured) — current ckpts fine; RGB/rgbd policies (h_rect, a3r, Dennis) affected.
Dennis should train on `human_fullpp_rgbd(_eef)` unmodified once the LUT lands.

## 4. Side studies

- **Gaze** (`datasets/human_gazerect`, 16 GB): full-res 2560×1920 rectified frames +
  30 Hz eyegaze + poses + row-aligned tables. Gates: 135/135 rows OK, align ≤16.5 ms,
  in-bounds ~100%; vergence depth UNRELIABLE (median 0.38 m vs scene 1.11) → 3-D gaze
  uses need ray-∩-stereo-depth (queued). Probes: scratch CNN 8.49° ≈ frozen DINOv2
  8.71° on 27-ep val (train-mean 13.0°) — two encoders hit the same wall ⇒ suspect GT
  parallax noise (up to ~4.7°) and global-pooled heads; next lever = GT refinement,
  then spatial heatmap head. Code/caches: tmp/gaze_probe*.py, gaze_cache_{train,val}.npz.
- **Camera-stabilized 3D flow** (Track4World/egowam): all 135 clips done, 130/135 pass
  gates (`egowam_out/GATES.csv`); 10,501 usable pairs; failures = 4 nav clips
  marginally >12 mm + `unknown_...165754` (also failed here AND gaze-probe worst —
  exclude that recording from future training).
- **Monocular metric depth** (for pretraining / no-stereo corpora):
  PointDiT-B = scale-normalized, unusable absolute (AbsRel 0.795, scale 4.9±1.3).
  **MetricAnything student-depthmap** (RGB + f_px): raw AbsRel 0.227, scale
  **1.171±0.023 (stable!)**, aligned 0.187 — near-absolute up to ONE constant.
  Verdict: below policy-grade (knife-edge), OK for PRETRAINING with ×1.171 calibration
  + real-depth fine-tune. Inputs were blurry 224→512; native-512 rerun would tighten.
  Repos: `scratch/pointdit` (env `pointdit`; DINOv3-B weights CONVERTED from ungated
  timm mirror, bit-exact, incl. hub-cache gotcha: loader caches by filename in
  `~/.cache/torch/hub/checkpoints` — replace THERE), `scratch/metric-anything`
  (must run from `models/student_depthmap/`; f_px=308.052 for our 512 rect grid).

## 5. Visualization tool suite (all reusable, all in tmp/ unless noted)

- **Interactive dual-stream viewer (artifact)** — the "what does the policy actually
  eat" page: two orbit panes (global glass + local eef ball), true colours, eef star,
  slider/play. Template `scratchpad/final_inputs_viewer.html` (self-contained,
  `__DATA__` inject), exporter inline in session (quantized int16 mm + u8 rgb + jpeg
  refs from `dp3c_mix186` val slot 19 = old ep23). Published:
  https://claude.ai/code/artifact/ff7a68f2-7d55-46fe-8436-4a523411fdfd
  (older full version with DP3 critical-point influence + input-impact panels:
  `scratchpad/dp3c_viewer.html`, data via tmp/export_viewer_data.sbatch).
- **xembody_chunkviz.py** — robot teleop frames → human policy → predicted 32-step
  eef trajectory FK-rolled and drawn on RGB + cloud panels vs teleop label
  (CKPT/TAG/EP/NEUT env; mp4+png). THE tool for "are the teleop labels even good".
- **o3d_real_clouds.py / o3d_depth_compare.py** — headless Open3D sphere-style
  renders (EGL on GPU nodes; solid matte = defaultLit + roughness 0.95 + indirect
  18k; traps: normals once on expanded layout, park empties at anchor,
  `setup_camera(fov, center, eye, up)` signature). Real dp3c data and
  Fast-FS-vs-monocular comparisons; upright rotation pitch −42°, oblique camera.
- **pointdit/metricany eval harness** — tmp/pointdit_run2.sbatch, metricany_run.sbatch
  (AbsRel/scale/aligned vs any episode's Fast-FS GT).
- Older: gaze videos/curves, flow GATES.csv + per-clip mp4s, dp3c arch figures.

## 6. Ops notes added this round

- SLURM: REQUEUE_HOLD "launch failure limit exceeded" = system hold, `scontrol release`
  DENIED for users → scancel + resubmit (resume-from-last.ckpt makes this safe).
  Controller outages recur (two multi-hour dual-controller downs); guarded-submit
  loops with existing-job checks are the pattern (tmp/flow_submit_guard.sh).
- NFS: freshly-written files can be served STALE to compute nodes for ~seconds-minutes
  (two incidents: engine.py patch raced a 1-s-later job; torch-hub cached the
  incomplete weight file). Leave ≥1 min between patch and submit; beware
  load_state_dict_from_url's filename cache.
- The classifier gate blocks writing cloud credentials (~/.aws) — user must run
  `aws configure` themselves if the public-EgoVerse download is ever revisited
  (keys are in origin/main README; recommend rotating them — they're public).
- Public EgoVerse (arXiv 2604.07607) = origin/main of THIS repo; zarr episodes via
  R2 `rldb` bucket + SQL table; NOT pursued locally (user deferred).

## 7. Standing queue

1. v5/v6/v7 finish (~5–7 h compute remaining each) → trim + neut verdicts →
   fold into leaderboard; then deploy picks for the next hardware session.
2. Hardware: v2/v4 vs old dp3c_dual same-start A/B (contract shipped); LUT
   regeneration confirmation from the deployment agent.
3. `--subject_scale` sweep on Chendong when the patch lands.
4. Gaze GT refinement (ray-∩-depth) → re-score both probes → maybe heatmap head.
5. MetricAnything native-512 rerun if pretraining route is pursued.
6. Height caveats message from user never re-pasted ("consider limb" — presumed
   arm-length-based scaling; reach-estimation pass designed, not run).

---
## 8. AUDIT 2026-09-12 (self-directed, no new training)

**Verified sound** (independent re-derivation, not trusting build logs):
- Split: loader's seed-42 shuffle == assembly placement; val slots hold ONLY old
  episodes; train∩val old-episode overlap = ∅; old-27 leaderboard ⊂ new-36 val (27/27).
  → every dp3cmix number is uncontaminated.
- All 5 mix datasets: row/episode counts match info.json exactly, 0 non-finite,
  rot6d orthonormal to 1e-7, eef-ball radius max 1.50 m with 0 violations and the
  <64-pt fallback never triggered.
- **Neutralization radii confirmed by radius profile**: grey fraction is 1.00 inside
  each dataset's own radius and drops immediately after (n15 →0.63 at 0.20; trim/0.30
  →0.53 at 0.45; n45 holds 1.00 through 0.45). The three variants are genuinely
  different data.
- Trim physically applied: 37 episodes shortened, 2054 rows removed, lengths differ
  per-episode in episodes.jsonl.

**Finding — the toy trim is weaker than the old corpus's raise-cut.**
Apples-to-apples on the same wrist-to-glasses signal:
| corpus | kept-tail closest approach | frac <0.40 m | episode fraction removed |
|---|---|---|---|
| old 135 (2026-07-19 rule) | 0.429 m | **0.12** | med 23–27%, 135/135 demos |
| toy, untrimmed | 0.223 m | 1.00 | — |
| toy, my trim | 0.41–0.45 m | **0.43 / 0.20** | med ~11%, 37/51 demos |
Cause is NOT the >50%-guard (blocked 0 demos) and not the tail-skip: the strongest cut
under my terminal-contiguous rule is only 12–14% of an episode. The old rule's name
records a **"steep-descent walkback"** component I did not implement — it removes the
approach ramp, not just the contiguous <0.40 run. → v5/v6/v7 carry ~⅓ of the intended
trim; treat their trim effect as a LOWER BOUND. A corrected cut (add descent-ramp
detection, target kept-tail ≥0.43 m / <0.15 frac) is the cheap next data fix.

**Measurement traps found in my own probes (do not repeat):**
- eef *height* in the glass frame is NOT a hand-raise proxy (glasses move with the
  head; retargeted eef ≠ human wrist). Use wrist-to-head distance from `*_aria60.npz`.
- max-radius over ALL grey points measures colour-invalid points, not neutralization;
  use the grey-fraction-vs-radius profile.
- bbox diagonal ≠ diameter: a filled 1.5 m ball gives ~2.6–3.5 m diagonal and will
  false-alarm the deployment guide's "3.0 m diameter" gate. Gate on max radius from
  the eef instead.

**Fleet final numbers (all 7, shared-27 serving-path MAE, final ckpts)**
v4_h128(first-32) 0.0815 < v2_drop07 0.0821 < v3_2048 0.0843 < old dp3c_dual 0.0847 <
v5_trim 0.0851 = v6_trim_neut15 0.0851 < v1_base 0.0868 < v7_trim_neut45 0.0888.
Reading: trim/neutralization variants do NOT beat v2/v4 offline — but v5/v6 vs v1
(same recipe ±trim) is +0.0017 WORSE on this val, which is expected and not damning,
because the val episodes are old-corpus (already cut) while the trim only changed toy
training rows; the trim's payoff is a hardware behaviour (no end-of-task raise), not
an offline metric. **Neutralization radius: 0.15 ≈ 0.30 ≫ 0.45** — the largest radius
is clearly worse (greys out task-relevant scene), the smaller two are indistinguishable
offline, so prefer 0.30 (existing deployment contract) unless hardware says otherwise.

---
## 9. LIVE WORKING STATE (2026-09-14 evening, pre-compaction snapshot)

### Running trainings (all 2000 ep, h=64 per the new standing rule: TRAIN 64 / deploy 32)
| run | job | data | delta vs v8 | state |
|---|---|---|---|---|
| v8  | 3823550 | dp3c_mix186_v2 | baseline of the v2 wave | ~ep600, ~100 ep/h |
| v9  | 3823724 | dp3c_mix186_v2 | + LEFT-eef cloud (front_pcd_3 / obs.aria_pcdc_local_left) | ~ep500 |
| v10 | 3824681 | dp3c_mix186_v2aux | 58-D target = 49 action ++ 9-D eef pose (aux prediction) | just relaunched, healthy |
| v11 | 3823725 | dp3c_mix186_v2 | proprio dropout 0.4 (noise 0.03 kept) | ~ep600 (was preempted+resumed, fine) |

**dp3c_mix186_v2** (71,823 rows): 1.0 m eef balls (was 1.5), NEW left-eef stream,
PROPER descent-ramp trim (14%/0% tails <0.40 m vs old-standard 12%), SG-smoothed
actions (w=9,p=2: −30% chatter, 0.08° cost — retarget was ALREADY 6× smoother than
teleop, measured). `_v2aux` = same + 58-D action column
`actions.joint_base_torso_head_arm_hand_eef`.

### Live watchers/loops (login-node bash, restart if dead)
- `tmp/diag_loop.sh` — submits `tmp/train_diag.sbatch` every 2 h; battery per ckpt:
  input-reliance ablations (proprio/global/local/left), per-block val MAE,
  std(pred)/std(GT) collapse detector, chunk |d²| jitter, v10 eef-err-cm.
  Outputs tmp/diag_RBY1_dp3cmix_v*.csv/.png. Reads ckpts only — safe.
- aux_chain / v2_chain: COMPLETE, no longer needed.

### Bugs fixed today (do not re-introduce)
1. v10 crashed at step 1: model `ac_keys.rby1` must be
   `actions_joint_base_torso_head_arm_hand_eef` (string-matched vs schematic).
2. `obs.aria_pcdc_local_left` existed in parquet but was MISSING from info.json in
   all 4 v2 datasets (patch anchor no-op) — fixed; also killed a stale v9 that had
   started pre-fix, and a duplicate v8 (chain + manual double-submit).
3. HARD RULE recorded: ≤8 CPUs per GPU on every sbatch (admin complaint); all 35
   templates in tmp/ patched.

### Answers established today (with plots in tmp/)
- Retarget actions are the SMOOTHEST source (|d2| 0.0008 vs teleop 0.009);
  action_curves.png / action_zoom.png. SG w=9 p=2 adopted for v2 data.
- NO x/y crop on the global stream — frustum-only (79.6° FOV); 31% of points >1 m
  off-axis (crop_xy.png). User declined? — not asked; option documented, unresolved.
- Crop criteria verified 0 violations; local = RIGHT-eef 1.5 m ball (v2: 1.0 m).
- v5/v6/v7 leaderboard (shared-27): v5 0.0851 = v6(n15) 0.0851 < v7(n45) 0.0888;
  neut 0.45 clearly worse, 0.15≈0.30. Full 7-fleet table in §8.

### Handoffs delivered
- `ai_docs/TRAINING_DATA_SANITY_HANDOFF.md` + `assets_rect_lut/mix186_manifest.csv`
  (per-episode slot→corpus→demo→recording→VRS) — for the other agent's visual data
  audit; §6 raw-data/pipeline context was given to the user as chat text to append.
- Interactive final-inputs viewer artifact:
  https://claude.ai/code/artifact/ff7a68f2-7d55-46fe-8436-4a523411fdfd

### Next actions when resuming
1. Check v8–v11 finished (ETA: v8/v9/v11 ~afternoon 09-15, v10 ~evening); release/
   resubmit on REQUEUE_HOLD (scancel+resubmit — release is admin-only).
2. Run leaderboard eval (pattern: tmp/dp3cmix_eval567.sbatch; v10 needs first-49-dim
   scoring — its 58-D MAE is not comparable).
3. Read diag CSVs — v11's d_proprio trend answers the leakage question; v10's
   eef_err_cm answers whether aux helps.
4. Await: other agent's toy_raise_cuts.json (their cut table — cross-check mine),
   data-sanity visual audit results, hardware A/B (v2/v4 vs old dp3c_dual), LUT
   confirmation, --subject_scale patch.

---
## 10. BASKET ROUND + collision findings (2026-09-15 overnight)

**New collection `chuye_basket`** (60 VRS, Chuye ~160 cm, quick basket demos med 8.6 s,
sanity 60/60 OK, hand-cov ≥0.95). Goal: ONE policy, original dp3c_dual recipe
(1.5 m ball, 0.30 neut, 1024 pts, dropout 0.9, NO SG) + standing rules (h=64 train /
32 deploy; proper raise-trim with NEW end-guard). Split: first 50 capture-order =
train, last 9 = val (59 demos — rec 4 dropped, EgoPoser singular rotation),
placement-controlled, valid_ratio=0.16. Run: RBY1_dp3c_basket/basket60_r1, 2000 ep,
deploy last.ckpt. Chain: tmp/basket_chain.sh (+Monitor) drives retarget→lerobot→trim→
rgbd→merge→dp3c(4-way)→assemble(datasets/dp3c_basket60)→train, all guards+logs in tmp/.

**Findings (cross-verified with audit agent, both sides converged):**
- TAGGER COLLISION (vrs_to_all.sh:85, loop-counter fallback): unnumbered
  Chuye_Basket VRS collided with _60 → demo duplicated + one recording lost. FIXED
  for basket by renaming the VRS to _1_ + full rebuild. SAME BUG in Chendong_toy:
  Chendong_0907_21 recorded twice → demo_19==demo_20 byte-identical, recording
  143630 lost — present in ALL dp3c_mix186* (v1–v11 train data). Queued fix (not
  applied — fleet mid-training): rename second _21 VRS (e.g. →_31), rebuild
  Chendong_toy_hdf5 + toy sets + one clean mix, retrain.
- µs/ns TRAP: aria60 npz ts is MICROSECONDS; ×1000 == aria_ts exactly, row r↔frame r,
  gaps trailing-only. Audit agent's "trim index-space bug" withdrawn; their
  trim2_basket_fixed.json DISCARDED. My trim tables (basket + toys) are correctly
  aligned; 1 real basket miss (demo_15) fixed by END-GUARD (kept rows may not END
  <0.40 m, floor 0.5·N) — now in tmp/basket_trim.sbatch with a hard unit-assert gate.
- CHENDONG SCALE BIAS measured in mix v2 data: median eef reach 0.587 m vs old
  0.628 / chuye 0.651 (6–10% compression, 20% of train demos); toy torso RMS
  0.045–0.074 vs old 0.118. Real but moderate — does NOT explain "very bad" hardware.
- VERDICT on bad hardware results: data defects predict mild symptoms (short reach,
  end-of-task raise for v1–v4 ckpts). Prime suspects remain serving-contract
  mismatches (v4=128-step chunks must slice to 32; v3=2048 pts; v6/v7 neut 0.15/0.45)
  — need from hardware: WHICH ckpts + failure description/video. Toy-mix "should be
  better" was never supported offline (v1 0.0868 vs old 0.0847 on identical val).
- Also queued: EgoPoser guard for singular hand-tracking rotations (recover
  recordings like basket rec 4 / future crashes).

---
## 11. ABLATION CAMPAIGN (launched 2026-09-19, weekend autonomous run)

**Goal**: single-variable ablations of dp3c_dual on chuye basket data. V0 =
RBY1_dp3c_basket_v2 (already trained, ep1999). Plan doc = the A–J table the user
pasted (also scratchpad dp3c_ablation_plan.md). Inference-time (K) excluded.

**Infrastructure (all in tmp/abl/)**:
- `manifest.tsv` — 25 runs, priority-ordered: 14 CFG (G1,G4,G6,G7,H3,H4,H5,J1,J2,
  J3=batch64,J3b=batch128,J4=lr3e-4,J4b=lr3e-5,J5) + 11 DATA (A1,A2,A3,A4,A5,B2,C3,
  C4,D1,D3,E4). Overrides per run in `ovr/<name>.txt` (Hydra CLI, incl. ~deletions
  for G4/G6/G7).
- `build_variants.py/.sbatch` — ONE 4-stripe pass computes ALL 11 DATA-variant clouds
  from the shared base cloud per frame (npy/<VAR>/). `assemble_variants.sbatch`
  swaps cloud columns into V0 parquets → datasets/dp3c_abl_<ID> (identical rows/
  split/actions to V0; only the stream differs).
- `queue.sh` (login node, PID in ps) — keeps ≤7 `abl` trainings queued, submits when
  dataset exists, retries ≤3, flags CFGFAIL on config crashes. `queue.log`.
- `eval_loop.sh` → `eval_final.sbatch` — scoreboard on the 8 val episodes,
  THE 3 METRICS: mae32, reliance deltas (d_proprio/d_global/d_local), jit_ratio.
  → `tmp/abl/results.csv`. Obs built from each ckpt's own schematic (handles
  G4/G6/G7 missing keys, D1/D3 shapes).
- wandb project **dp3c_ablation** (personal entity), run id = RBY1_abl_<ID>_abl.

**Deferred (explicitly, notify user)**: F1–F4 (need deploy-side IK), G5 (FK keypoint
proprio), H1/H2 (encoder code), I1 RTC / I2 flow-WM / I3 DINO-WM (code-heavy; attempt
only if waves stable). I5/I6/E1/C1/C2/D2 already settled.

**Cleanup done (user-granted)**: old-background toy derivatives deleted
(toy_*_dp3c{,_2048,_n15,_n45}, lerobot_raw, tmp npys/rgbd shards, ~12 GB); KEPT
toy_*_dp3c_v2 + toy_chuye/toy_chendong (mix-v2 rebuild capability).

**User expectation**: hardware testing Monday; pull latest ckpts even if <ep1999.
