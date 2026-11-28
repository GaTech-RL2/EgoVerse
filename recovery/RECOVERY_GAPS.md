# Recovery gaps relative to the last recorded Skynet work

Assessment date: 2026-11-27 (America/New_York). Assessed recovery revision:
`262fd6d406e383fe70b57f1ad10d94e9675b7f8a`, branch `rby1_local_chendong`.
GitHub baseline: `rby1_aria_shared@8663e4c9f62f6f6b620f60263313c0a773dc52be`.

The model/training core is substantially recovered. The largest gaps are the
complete latest preprocessing dependencies, processed training datasets, and
latest weights/full training state. This branch alone cannot reproduce the
latest raw-recording-to-training workflow or resume the latest Skynet runs.

**This is a comparison with retained records, not a live Skynet diff.** No
Skynet connection was made for this assessment. November 24–26 reports describe
past observations; later changes and current remote survival are unknown. A
reliable percentage recovered, total missing bytes, or complete missing-file
list cannot be calculated without a full matching remote inventory.

## How to interpret status

- **Present in branch:** source or metadata is actually tracked here. This does
  not imply its remote dependencies or referenced datasets are present.
- **Local only:** files were inventoried on the Mac outside Git. A fresh clone
  will not contain them.
- **Partial / not recovered:** some source, configuration or reports exist,
  but the complete named component or payload is absent from this recovery.
- **Unverified parity:** a usable version exists, but equality with the latest
  remote version has not been demonstrated.

“Not recovered” does not mean permanently lost. Other machines, archives or
backups may still hold it. Files excluded from Git are distinguished below from
files not found in the local recovery inventory.

## Code gaps

| ID | Component and status | What we have | What is still needed and why |
|---|---|---|---|
| C1 | Latest fixed retarget and Base Smoothing S — **partial / not recovered** | Invocation scripts, CPF/measured-wrist settings, audit parameters and source-hash evidence | Complete matching `SEW-aria-pipeline`, `aria_egoposer` handoff, EgoPoser source/weights and canonical `base_fix_1124` implementation. Calls and reports cannot reconstruct the exact implementation; end-to-end VRS-to-fixed+BS reproduction is blocked. |
| C2 | Latest HD2/depth/point-cloud pipeline — **partial / not recovered** | Earlier Nov21 RGBD/cloud/depth scripts and later wrappers | Complete shared `real_dp3c_tri_gz_vis_hd2_fix1124_v2.py`, its matching dependencies, model assets and calibration inputs. Earlier files do not establish the latest geometry behavior. |
| C3 | All latest training runners and performance patches — **unverified parity / partial** | Nov25 FT template integrated as `campaign/train.py`; strong MSC, GeoAug, LoRA; earlier Cloth and RGB variants | Exact per-run scratch/pretraining runners, environment wrappers and all later patches. For example, retained notes describe a 256-entry mmap cache/16,384 FD setup; complete latest per-workspace parity is not established. Configs for scratch180k do not themselves supply its exact runner. |
| C4 | Continuous EgoVerse preprocessing orchestration — **partial / not recovered** | Historical setup/recovery helpers and selected monitoring records; Drawer-specific controller scripts | Complete latest EgoVerse controller/worker versions plus the corresponding live ledger, attempts, submission intents, receipts and QC state. Drawer helpers are not a replacement for the multitask controller. Replaying old scripts could duplicate work or misclassify accepted items. |
| C5 | Production environment and auxiliary assets — **partial / unverified parity** | Local Python 3.12 lock, baseline environment files, robot model/configuration and deployment client source | Exact CUDA/BF16 environment, external model revisions/weights and package-specific geometry/gaze assets. `external/openpi` is an uninitialized submodule; the tested PCD path does not use it. Mac CPU verification does not establish full production readiness. |

Evidence: [retarget invocation](snapshots/cart-drawer-nola-fix-bs-20261125/drawer_retarget.py),
[BS reference audit](snapshots/cart-drawer-nola-fix-bs-20261125/bs-pdf-reference-audit.json),
[earlier preprocessing source](snapshots/cloth-new2-preprocess-20261121/),
[Nov25 FT template](snapshots/cart-drawer-nola-fix-bs-20261125/template/),
[adaptations](adaptations.json), and [retained task excerpts](gap-evidence.json).
Remote source locations recorded by these files include
`/coc/flash7/czhang883/Documents/SEW-aria-pipeline`, the
`mobile_generalist/data/aria_egoposer_warp_handoff/` tree, and
`/coc/flash7/czhang883/warp-data/pipelines/base_fix_1124/`.
These are recovery locators, not verified accessible paths.

## Data gaps and local holdings

No `.vrs`, `.pt`, `.pth`, `.ckpt`, `.hdf5`, `.h5`, `.npy` or `.npz` files are
tracked in the assessed branch. The branch contains code, configurations,
manifests and reports; payloads must be recovered or supplied separately.

| ID | Data category | Available evidence / local holdings | Gap |
|---|---|---|---|
| D1 | Mac raw Aria recordings — **local only** | Main staging has 352 distinct VRS files, 70,925,529,829 bytes (70.93 decimal GB), including all Cloth new2 takes 1–170 | Not included in this Git branch. Coverage of the complete Skynet raw collection, including EgoVerse source data, is unknown. The 352 files must not be interpreted as a full remote backup. |
| D2 | MPS — **local only, incomplete** | Main staging contains 5,636,274,191 bytes (5.64 GB) of MPS; 165 Cloth takes have SLAM/hand/gaze components | Cloth takes **123, 132, 152, 157, 160** lack MPS in the recorded inventory, although their raw VRS files are present. Full MPS coverage for every other task is not established. |
| D3 | Latest fixed/BS target releases — **not recovered as complete payloads** | Cart/Basket/Cloth/Drawer manifests, split/QC reports, migration records and some normalizers | Complete retarget outputs, action/state tensors, RGB/depth/point clouds, DINO features, timestamps/masks, camera/calibration arrays and all referenced shards. A retained `READY.json` or hash is not the referenced data. Local samples or older artifacts do not close this gap. |
| D4 | EgoVerse source/pretraining releases — **not recovered as complete payloads** | Historical progress and audit records | Full source recordings, processed Pick/Bag/Fold/Object data, train/validation membership and version-matched normalized releases. Latest remote processing was not yet recorded as a final READY release. |
| D5 | Canonical publication/Atlas tree — **metadata only in branch** | Four-task migration and Cloth165 publication reports | Actual collection contents, versioned dataset releases, referenced file targets and complete registry/provenance. Local paths must be reconciled; archived remote symlinks/paths do not become usable through cloning. |
| D6 | Cloth last105 fixed+BS release — **pending in last retained record** | Raw/MPS selection and a 95-train/10-validation plan before QC | The handoff awaited Chuye's processed release. There is no basis to claim a completed latest release or its planned FT60k/scratch180k weights existed at that point. This is pending work, not a verified lost completed artifact. |

The Mac inventory root is
`/Users/chendongxin/Research-RL2/aria_gen2_scripts/data`. Its total staging size
was 76,939,924,231 bytes after hardlink deduplication. Header reads established
local availability, not full playback or checksum validation. Downloads,
external/cloud-only storage and equality against Skynet were not fully audited.
Inventory counts are dated observations, not a fresh full data validation.
See [inventory evidence](gap-evidence.json),
[Cloth165 publication record](snapshots/cloth-new2-canonical-20261125/README.md),
[Cart release report](snapshots/bs-fixed-target-scratch180k-20261125/cart/releases/cart/READY.json),
and [migration records](snapshots/warp-data-consolidation-20261125/).

The last retained Nov26 EgoVerse controller record reported 1,428 worker-accepted
items; the last independent count remained 1,243 because a Fold artifact marked
ACCEPTED had a conflicting Slurm NODE_FAIL result. Neither number is a count of
locally recovered data, and neither proves a complete training release. Current
remote progress is unknown. The original ledger and audit discrepancy must be
reconciled before using these records as recovery acceptance evidence.

## Weights and exact training resumption

Older Nov19/20 deployment weights exist locally outside Git. The restored code
was checked against the Nov19 Basket B10k package: strict loading of 893 state
keys and three plain/RTC action comparisons passed. That validates an older
deployment, not the latest Nov25–26 weights or optimizer state.

The following newer artifacts have retained remote evidence but **no verified
complete matching local payload in this recovery**:

| Artifact | Retained locator/evidence | Missing recovery material |
|---|---|---|
| Cart107 fixed+BS scratch180k | [completion record](snapshots/cart107-bs-scratch180k-deploy-20261125/complete.json): `cart107-la-fix1124-bs-scratch180k-MSCstrong-robotpc-20261125`, step 180,000 | Matching package/weights; source checkpoint SHA starts `28ef10b10840`, portable checkpoint SHA starts `99a8804d695e`. The report's package size is 597,110,500 bytes; it is not a local inventory or total training-state size. |
| EgoVerse180k → Cart107 fixed+BS FT60k | [completion record](snapshots/cart107-bs-ft60k-deploy-20261125/complete.json): `egoa180-cart107-la-bs-FT60k-robotpc-20261125`, step 60,000 | Matching package/weights; source checkpoint SHA starts `e851f93e7d43`, portable checkpoint SHA starts `3668bd500803`. |
| Shared fix1124 Cart/Basket FT10k and Cloth deployments | [Cart/Basket package manifest](snapshots/sharedfix1124-ft10k-deploy-20261124/reports/packages.json), [Cloth completion report](snapshots/egoa-target-retarget-fix-20261124/deploy-cloth-ft10k-complete.json) | Matching checkpoint, gaze ensemble, virtual camera/geometry and normalizer files for each package. Recorded hashes identify targets but do not supply their bytes. |
| Other latest pretraining/FT/scratch/LoRA/GeoAug runs | [scratch configurations and reports](snapshots/bs-fixed-target-scratch180k-20261125/), [LoRA setup](snapshots/cart-drawer-nola-fix-bs-20261125/setup_lora.py), [GeoAug configuration](snapshots/cart107-nola-geoaug-20261125/proposed-config.json) | Exact latest checkpoint per run and complete resumable state. Final step/completion must remain unknown where only a launch/configuration or earlier progress record is retained. |

For exact resumption, recover all state required by the original runner:
model/EMA where used, optimizer, scheduler/scaler where used, RNG state, step and
sampler state where stored, configuration, code identity, normalization and
dataset hashes. An inference-only portable checkpoint is insufficient. This
recovery deliberately changes the source layout/code hash; never substitute a
fabricated old identity to bypass resume validation. See
[checkpoint verification](checkpoint-validation.json) and [adaptations](adaptations.json).

## Recovery priorities and closure criteria

1. Preserve available Mac raw/MPS data and older deployment packages in a
   separate verified backup. Keep large payloads outside Git; retain per-file
   checksums and recording IDs. Resolve the five missing MPS takes separately.
2. Recover C1/C2 upstream source and model assets from another verified copy or
   backup. Record exact revisions plus uncommitted changes; validate one known
   recording against retained retarget/FK/geometry evidence before bulk rebuilds.
3. Recover the selected processed releases and newest weights as matched sets.
   Validate all manifest references, hashes, splits, dimensions and normalizers;
   do not combine similarly named releases from different geometry versions.
4. Recover exact per-run runner/environment and controller/ledger state if
   continuing the original experiments. Reconcile attempts and terminal states
   before resuming work; historical scripts should not be run blindly.
5. Run real-data loader checks, CUDA/BF16 smoke/resume tests and deployment
   comparisons on suitable hardware. CPU synthetic tests alone do not close the
   production or hardware acceptance gaps.

The canonical future data destination remains
`/coc/flash7/czhang883/warp-data/`: raw inputs under
`collections/<collection-id>/raw/`, processed outputs under
`collections/<collection-id>/processed/<run-id>/`, versioned training/visualization
releases under `datasets/<dataset-id>/releases/<release-id>/`, and build provenance
under `pipelines/`/`registry/`. This describes the agreed layout; no publication,
relocation or Atlas visibility check was performed during this assessment.

Close a gap only when the matching bytes/source are available, provenance and
hashes are recorded, and the relevant checks pass. Update this document with
the artifact location, revision/hash and validation result. Preserve unresolved
remote parity as unknown until it can be checked.
