# EgoVerse local recovery — 2026-11-27

This version starts from GitHub branch `rby1_aria_shared`, commit `8663e4c`
(2026-10-27), and incorporates November deployment sources, training templates,
and subsequent fixes retained on the Mac. Recovery did not connect to Skynet.
This branch supports continued development and local verification; byte-for-byte
equivalence with the last inaccessible Skynet workspace has not been established.

## Code entry points and provenance

| Location | Recovered content | Source and limits |
|---|---|---|
| `egomimic/` | Patch/transformer point-cloud encoders, segmented sampling, HPT/flow and training helpers | Nov20 deployment package compared file by file with GitHub; baseline files absent from the snapshot retained |
| `campaign/policy.py`, `hd2_model.py`, `objectives.py`, `kinematics.py` | A/B/C/D objectives, DINO future head, FK, policy transfer | Nov19 PCD deployment package; policy matches the Nov25 template |
| `campaign/data.py` | 49-D actions, 22-D proprioception, H64, DINO +10, tail masks, hashes and mmap | Complete source retained in the Nov25 template |
| `campaign/train.py`, `contracts.py` | Fixed-target B fine-tuning, 60k steps, original full-state checkpoint and acceptance requirements | Nov25 FT template with same-day Drawer name support restored |
| `campaign/photometric.py` | Parameterized strong MSC and optional GeoAug | Nov19 source plus explicit Nov23/25 patches from local history |
| `campaign/geoaug.py` | Point dropout, metric jitter, shared grayscale | Complete Nov25 local source; enabled only through configuration |
| `campaign/dino_lora.py`, `train_lora.py` | Inherited/frozen DINO head with rank8/alpha16 LoRA; full FT for other parameters | Nov25 module source and replayed `setup_lora.py` patch |
| `campaign/train_cloth.py`, `contracts_cloth.py` | Earlier Cloth PCD scratch/FT runner | Complete Nov21 version; latest Nov25 scratch runner not established |
| `campaign/train_rgb.py`, `rgb_policy.py`, `rgb_data.py` | Separate RGB variant | Nov20 RGB deployment package, separate from the PCD runner |
| `configs/hd2-reference.yaml`, `assets/robot.*` | PCD architecture and RBY1/XHand FK model | Nov19 deployed package |
| `scripts/fit_normalizer.py` | Source-train-only quantile statistics and constant-dimension guards | Retained Nov08 source; does not replace all latest target-only/pooled adaptation recipes |
| `deployment/robotpc/` | Deployment support and background-observation/RTC clients | Nov19/24 local files; no robot connection tested; paths still require deployment-package configuration |
| `recovery/snapshots/` | Preprocessing, fix1124, BS, Cart/Drawer, Cloth recovery, LoRA, GeoAug, data migration, recent configurations and acceptance evidence | Historical files grouped by source; not a directly runnable local pipeline |

Original sources and SHA-256 values are recorded in `import-manifest.json`;
explicit recovery changes are recorded in `adaptations.json`; current file hashes
are in `file-manifest.json`. Duplicate import records remain in the provenance;
the actual file uses the last documented source. All original Mac data, source
snapshots and deployment packages remain in place.

Documentation and human-readable metadata are English. The English-only update
also translates two archived audit records and inherited CAD component labels.
These exceptions to exact snapshot preservation are documented in
`adaptations.json`; original import hashes and the preceding Git revision retain
their provenance. Numeric data, mesh paths and executable logic are unchanged.

## Verification on the Mac

```bash
cd /Users/chendongxin/Research-RL2/warp/EgoVerse-local-recovery
.venv/bin/python -m recovery.verify
```

An isolated Python 3.12 environment has been created. To rebuild it:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-local.lock.txt
```

`requirements-local.lock.txt` records exact package versions in the local
verification environment. It does not reconstruct the missing Skynet CUDA
environment lock. Original `requirements.txt` and `environment.yaml` are retained.

The verifier checks unit behaviors, A/B/C/D CPU forward/backward/updates, FK
against independent kinematics, paired initialization/transfer, gaze and runner
CLIs. Model access during verification disables Hugging Face downloads; the
checks do not start WebSocket services, training jobs or robot connections.
Results are written to `recovery/runtime/validation.json`; the initial recovery
results are retained in `validation.json`.

The existing deployment package also supports strict checkpoint loading and
comparison of three plain/RTC action cases against its CPU references:

```bash
.venv/bin/python -m recovery.verify --checkpoint-package \
  ../SEW-Geometric-Teleop/docs/ai/artifacts/basket-ft10k-deploy-r2-20261119/basket_B180_MSCaug_FT10k-robotpc-20261119-r2
```

This checks the **Nov19 Basket B, 10k checkpoint**, not the latest Nov25 training
weights. Weights are kept outside Git. The model has 68,101,864 parameters and
outputs actions with horizon 64 and dimension 49.

## Training and data status

Recovered production runners still require CUDA/BF16, real releases,
normalizer/manifest checks, smoke tests and the applicable acceptance gates.
These requirements were retained during Mac verification. `code_hash()` now
uses this repository's root-level `egomimic/`, ensuring source is counted instead
of looking under the old `source/egomimic/` layout. The resulting code identity
differs; it must not be presented as the old identity to resume training.

Saved Nov25 Cart/Basket scratch180k configurations, strong MSC, GeoAug and FT
recipes are under `snapshots/`. Their states and remote paths are historical.
No new READY status was produced, historical tasks have not been reclassified
as currently RUNNING, and old reports do not establish full QC for this version.

Local Aria staging contains 352 distinct VRS files totaling 70.93 GB, including
all 170 Cloth new2 takes, plus 5.64 GB of MPS data. Of the Cloth takes, 165 have
SLAM/hand/gaze files. Raw VRS files remain available for takes
123/132/152/157/160, whose MPS files are missing. Machine-specific locations are
recorded in Git-ignored `local-paths.json`; recovery did not relocate raw data.

## Incomplete recovery areas

See [Recovery gaps relative to the last recorded Skynet work](RECOVERY_GAPS.md)
for the detailed code/data/checkpoint inventory, evidence, status definitions
and recovery priorities. The list below is a short overview.

1. **Nov24 fixed-retarget upstream:** the complete `SEW-aria-pipeline`, EgoPoser
   handoff source and weights, and exact latest `aria_egoposer`/Base Smoothing S
   implementation were not found in the GitHub baseline or local snapshots.
   Calls, parameters (including CPF/measured wrists) and acceptance evidence are
   retained. No speculative implementation was substituted. End-to-end local
   VRS-to-fixed+BS release reproduction has not been established.
2. **Complete dependencies for the latest point-cloud/depth pipeline:** earlier
   HD2/cloud/RGBD sources and Nov25 invocation scripts are available locally, but
   complete upstream files such as the shared
   `real_dp3c_tri_gz_vis_hd2_fix1124_v2.py` and external models remain incomplete.
   Remote paths in historical scripts require recovery individually.
3. **Latest Nov25–26 full checkpoints and processed releases:** retained earlier
   deployment weights and data manifests do not constitute a complete backup of
   the latest pretraining/FT state, optimizer state or training data.
4. **Byte-level parity for all latest runners:** restored FT templates are
   traceable. Complete sources are still missing for other campaigns' separate
   scratch/pretrain runners, remote optimizations such as 256-mmap, and the
   continuous-processing controller. Earlier variants and newer configurations
   remain explicitly separated.
5. GitHub submodule `external/openpi` has not been downloaded; the PCD recovery
   checks do not require it. CUDA, complete real-data loading, image/depth
   reconstruction and robot rollout acceptance were not performed.

A historical log entry saying "passed" does not close these recovery gaps.
Further complete local sources should be incorporated with provenance hashes
and the relevant checks rerun.
