# Yaw scale investigation — 2026-11-26

Scope: offline, no deployment edits, inference requests, training submissions or robot commands.

## Verified evidence

The archived source normalizer SHA c2effb5afade0c889eead36ec6dff17ad51243544b0bd9b584325ea6a6b13389 matches the parent of the currently inspected Cart cone/ball FT packages. Their yaw parameters are unchanged. The November 17 data-audit.json independently recorded an exact source-train refit across 111,965 rows. Its audit.py implements the fallback below.

For each action dimension, fit q01/q99, then replace them with global min/max if `(q99-q01) < 0.05*(max-min)`. Add epsilon 1e-6 to width; scale is half width. The serialized quantile_1/quantile_99 fields contain the REPLACED bounds, so they cannot recover the true source quantiles.

Yaw (per-row delta, radians): source min -0.251384914, max +0.294711232, center +0.021663666, scale 0.273048580, sparse_fallback=true. Therefore its original central 98% interval was narrower than 0.027304807 radians (1.56445 degrees), and the fallback enlarged the scale by approximately more than 20 times relative to that central interval's half-width. This establishes tail-sensitive scaling, not that the tail labels are invalid.

Archived Cart target-only fits: November 7, 30 train recordings, q01=-0.010007027/q99=0.011567540, scale=0.010787783; November 8, 97 train recordings, q01=-0.009444447/q99=0.010612942, scale=0.010029195. Neither used fallback. Source scale is 25.31x and 27.23x larger respectively. These are older target data, NOT a measurement of the current BS FT dataset.

## Remaining uncertainty

Direct sky1/sky2 and sky1 via robotpc timed out; Atlas was unreachable. Robotpc has model statistics and small observation fixtures but not the complete source/current BS training action arrays. Consequently current BS raw quantiles, number of extreme rows, their recording identities, and whether they are true turns, wrap errors, time gaps or retarget discontinuities remain unverified. No claim that this proves clockwise drift.

## Proposed repair and required audit

1. Retrieve exact source and current FT manifests and SHA-verified action arrays. Report train and validation separately; fit on train only. Compare unmasked rows used by the historical fitter against action_valid rows actually supervised. Compute raw signed and absolute quantiles, per-recording extrema and contribution of extreme rows; retain timestamps and adjacent rows for inspection.
2. Check dyaw against the pipeline's actual frame convention and wrapped differences of heading, including segment boundaries and time gaps. Correct confirmed invalid labels/segmentation rather than deleting valid turning behavior.
3. Evaluate a dedicated yaw rule without the sparse-to-global-minmax fallback: zero-centered robust scale with an explicit physical floor, validated against legitimate turns and tail coverage. No numeric scale is selected without the current data. Do not blindly clip all tail actions or force scale to nearly zero on mostly stationary data.
4. Changing scale requires a consistently versioned training/inference normalizer and retraining/fine-tuning validation. Swapping only the deployed JSON changes the physical meaning of existing model outputs and RTC prefixes. For a lower-impact comparison, retain the old normalizer and test explicit physical-yaw error weighting during training; it is an experiment, not a proven fix. Flow-matching training/noise means simple output-layer rescaling is not guaranteed equivalent.
5. Offline acceptance should compare physical dyaw error, sign bias, 64-row integrated yaw, stationary false turns, valid-turn accuracy, and multiple sampling seeds on identical saved observations. Current rollout logs lack the exact historical predictions/RNG/prefix, preventing exact replay attribution.

Numeric provenance: yaw-distribution-archival-comparison.json. Historical refit proof: ../validation-audit-20261117/data-audit.json and audit.py.
