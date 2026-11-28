# Cloth new50: recovered fixed-data release

The new release preserves the original 43 accepted recordings and adds 17 disjoint, quality-accepted segments from 14 of the 15 erroneously truncated recordings. It contains 60 episodes and 24,577 rows, including 2,647 recovered rows. Original artifacts and ongoing training are unchanged.

- Data: `/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124/release/target.json`
- Training workspace: `/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124/ft`
- Run: `egoa180-cloth50-recovered-fix1124-ft60k-MSCstrong-s42`
- Unique Slurm job: `4855714`, normal A40; changed in place from pending L40S, no duplicate submission.
- Initialization: original EgoVerse B/strongMSC seed42 policy at 180k; fresh auxiliary DINO head, matching the existing FT recipe. Target budget: 60k, seed42, batch32, bf16, B + strong MSC.
- Split: original 38 training / 5 validation recordings; 21,891 / 2,686 rows. Every segment retains its original recording's split. Normalization remains frozen from source training data.

## Acceptance

All new accepted arrays passed finite/shape/hash checks, timestamp alignment, RGB correspondence, strict bilateral tracking, unchanged retarget/cloud quality gates, FK/clean-cloud tensor checks, and the actual H64/+10 loader. Segment boundaries are never stitched; the first action is masked when its base delta references the preceding gap. Old artifact hashes are unchanged.

All 17 new accepted segments were visually reviewed at first/middle/last frames using RGB, global/right/left clouds, and TCP/FK overlays. This is sampled visual review, not every-frame review. Original quality thresholds permit brief IK residual outliers; a sampled boundary error reaches 88.2 mm on take84, while robot-q-derived training FK tensors remain consistent. Local neighborhoods can be sparse outside camera coverage. These limitations are retained in `visual-review.json`.

Take94's additional interval failed the original FK gate and had no reliable >1s refinement. Its original accepted portion remains. Eight original failed candidate attempts are retained; seven have accepted stricter subsegments. These are candidate-attempt counts, not eight excluded recordings.

The independent smoke passed 100-step save → second-process resume to200, full validation of all2,686 rows, parent identity, optimizer/scheduler/RNG/sampler state checks, and finite updated weights. At step200 full validation action loss was0.216000 and auxiliary loss1.209212. Formal training starts separately from the EgoVerse parent, not from smoke weights.

Validation includes recovered frames, so its loss is not directly comparable to the old validation row set. See `release-summary.json` for per-recording recovered intervals and counts.
