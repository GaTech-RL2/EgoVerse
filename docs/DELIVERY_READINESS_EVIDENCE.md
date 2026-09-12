# Delivery and visualization evidence

Recorded 2026-09-12 for the v2.2 follow-up, scoped to vendor contribution (b)
and visualization (c). Training (a), including PR-29 completion, is deferred
until Sharpa supplies an eligible delivery. No GPU optimization was run.

## Contributions

| Design PR | Commit | Result |
|---|---|---|
| PR-30 | `63ac9ed6` | Loader and inspector calibration follow the selected image, including a different coordinate-reference camera and legacy front names. |
| PR-31 | `ea55c437` | Pose-only Human, raw Aria aliases and legacy intrinsics validate; existing raw/direct upload boundaries have isolated integration coverage. |
| PR-32 | `df953d08` | Malformed episodes produce individual reports; retained numeric values, hand/arm arrays, and explicit complete ego-delivery requirements are checked. |
| PR-33 | `13bfaeec` | Inspector and artifact modes share geometry, topology and masks; reduced dependencies retain RGB; horizon, status, annotations and 3D controls work together. Resized Aria exports retain matching intrinsics. |
| PR-34 | This documentation change | General, dexterous and preview instructions agree with these checks and distinguish remaining service/data gates. |

The retained Pi implementation changes were excluded from these commits. Dash
and its locked dependencies were added to the default environment so the public
inspector command works after `uv sync --locked`.

## Automated verification

Use the repository's normal CPU checks:

```bash
uv sync --locked
uv run --no-sync pytest --collect-only -q
uv run --no-sync pytest -q
uv run --no-sync ruff check egomimic
```

An export of the committed tree passed **295 tests**, independently of the
retained Pi changes. The working-tree suite passed **300 tests**, and repository
lint passed. The final inspector adjustment also passed its 10 focused
regressions. The default
manifest/lock were checked independently of the retained Pi extra. These are
local results; hosted CI and a fresh optional-Pi installation were not run.

Key regression coverage:

- `test_calibration.py` and `test_camera_coverage.py`: selected-image K and
  transforms, moving-camera horizons, legacy camera names, unavailable views.
- `test_legacy_compatibility.py` and `test_upload_compatibility.py`: legacy
  validation, actual Human/Aria and EVA uploader entry points, metadata naming,
  converters, ZarrWriter, processed-folder discovery, staging rows and ordinary
  local resolver loading. EVA uses a small HDF5 recording; only hardware-specific
  Aria extraction is substituted. S3 stores real bytes on local disk. The staging
  command executes against SQLite with PostgreSQL type/default syntax adapted.
- `test_delivery_validation.py`: complete-delivery prerequisites, malformed
  metadata in batches, retained/padded numerics, unit quaternions, joint limits
  and declared order, fixed cameras, exact RGB resolution, and masked slots.
- `test_dexterous.py`: supplied joints/keypoints agree with registered FK where
  available; wrong joint order or units fail. Missing optional models are reported.
- `test_inspector_compatibility.py`: inspector/artifact pixel parity, pose-only
  clips, actual raw-Aria edges, HTTP routes, camera/horizon cache keys, preserved
  missing 3D slots, and a fresh subprocess with embodiment imports unavailable.

The staging and resolver status guards both accept absent legacy status and
`complete`, and both exclude `structural_sample` and unknown statuses. These
guards establish status eligibility; they do not replace delivery validation.

## Browser and artifact review

The public local-folder invocation in [Delivery previews](DELIVERY_PREVIEW.md)
was exercised with Chromium, followed by visual inspection of screenshots and
beginning/middle/end rendered frames. The walkthrough searched and selected each
of five episodes, changed the horizon, switched all four overlay modes, toggled
annotations and 3D, verified trajectory/orientation/keypoint traces, played and
paused, and scrubbed to the retained tail. It finished with no browser page
errors. This is browser interaction plus image review, not a maintainer sign-off.

The walkthrough exposed a long-episode preload backlog. The inspector now limits
preloading to two requests at a time and cancels superseded requests. A second
visual check separated capability/status badges from annotation text.

Local artifacts are in `outputs/delivery-readiness/` (not versioned). Each MP4
has an adjacent JSON report; `browser-walkthrough.json` and
`browser-<episode>.png` record the browser checks.

| Artifact stem | Rendered / selected frames | Validation errors / warnings | Interpretation |
|---|---:|---:|---|
| `human-keypoint` | 4,100 / 4,100 | 0 / 5 | Existing canonical Human recording; reported established Aria calibration fallback. |
| `human-cartesian` | 90 / 90 | 0 / 5 | Same recording, future pose horizon of 16. |
| `sharpa-analysis-keypoint` | 370 / 370 | 4 / 2 | Existing estimated analysis copy; remains an ineligible structural sample. |
| `synthetic-dexterous-keypoint` | 12 / 12 | 0 / 1 | Explicit synthetic contract fixture with registered Sharpa hands, moving ego camera and `camera:left_wrist` reference. |
| `human-pose-only-cartesian` | 12 / 12 | 0 / 2 | Derived Human clip without keypoints; familiar trajectory mode works. |
| `human-missing-calibration-keypoint` | 0 / 12 | 0 / 4 | Derived negative fixture: RGB and annotations remain visible; projection reports missing intrinsics. |

Visual checks confirmed consistent side colors and recognizable finger topology
across the inspector and artifact renderer. Multi-frame geometry uses the
displayed frame's camera pose, with retained-tail clipping. The raw Aria alias
and non-front camera cases also have deterministic regression comparisons.

The Sharpa analysis copy now reports joint-limit violations for both hands,
three non-increasing RGB timestamp steps, and annotations/task text containing
metadata delimiters. Its camera/keypoints remain estimates. Rendering all 370
frames neither clears those errors nor demonstrates measured calibration.

The synthetic fixture passes the documented complete-delivery command with all
metadata flags and `--ego-overlay` (`synthetic-complete-validation.json`). The
ordinary `LocalEpisodeResolver`/`MultiDataset` loads its supplied arrays unchanged
at frames 0, 6 and 11, with finite values, five-frame horizons, repeat-last tail
padding and the selected front K (`synthetic-local-load.json`). This demonstrates
the input contract; synthetic geometry cannot certify a physical vendor capture.

## Remaining acceptance gates

| Criterion | Result and remaining gate |
|---|---|
| (b) Contribution infrastructure | Local validation and upload/conversion/staging/discovery checks pass. A service round trip remains pending: the maintainer knows of no designated nonproduction upload prefix or test database. No test data was sent to production. |
| (c) Visualization infrastructure | Browser workflow, shared geometry and local artifacts are exercised for Human, synthetic dexterous and estimated Sharpa data. Measured Sharpa delivery review remains pending receipt of compliant data. |
| (a) Training | Deferred. Eligible Sharpa data, pretrained optimization, legacy 32D regression, checkpoint continuation and fresh optional-Pi installation remain unverified. |

When isolated services are available, use the existing upload path with a small
legacy episode and a compliant dexterous episode, stage/register their metadata,
and load them through the ordinary service-backed resolver. Record the assigned
prefix/database, row fields and resulting load. This service gate is distinct
from the local substitutes above.

Sharpa can use the [dexterous contribution contract](CONTRIBUTING_DEXTEROUS_HANDS.md)
and [local visual checks](DELIVERY_PREVIEW.md) to prepare deliveries now. These
changes do not claim that all v2.2 release gates, or criterion (a), have passed.
