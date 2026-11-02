# Tracking-valid validation windows

Evaluation policy: `tracking_valid_windows_v1`.

Exact even-stride validation exposed finite missing-hand sentinels in the
transferred data. These can pass a finite-number check while producing enormous
losses. A missing current wrist also invalidates the relative-action coordinate
frame; masking only the final scalar error would still feed invalid conditioning
and target actions through the model.

The evaluation reader checks the existing raw missing-hand convention before
transforms and normalization. A selected start is unscorable if any required
current, actual history, or future action read is missing. It keeps that exact
episode/frame identity and does not replace the frame. The evaluator excludes
the whole unscorable sample before model execution and every scalar reduction.
Metrics use their actual surviving sample counts. All-invalid batches contribute
no fabricated zero score. Ordinary training reads, model settings, batch size,
normalization, random seed and padding policy remain unchanged.

Every cohort records visited/scored/skipped starts, both in total and per episode.
Every requested episode must have at least one scored start. Pinned videos retain
all source frames and their timing; unavailable tracking intervals are labeled
and carry no model prediction. Every video must contain a scored anchor, and its
unscored frame spans are recorded. Smoke and production completion verify this
evidence as well as existing loss, checkpoint, W&B and episode/video-count gates.

These scoring semantics differ from earlier unmasked runs. Compare the 36 cells
only under this policy; rescore older checkpoints under the same policy before
including their scores in a comparison.

Existing normalization artifacts retain their original producing commit and
checksums. Reuse after this evaluation-only source change requires an explicit
`normalization_reuse.json` compatibility receipt, an independent review of the
exact source diff, unchanged training/normalization kernel, and unchanged data,
all 36 splits/configurations/reports/statistics. The source check rejects model
or configuration changes and does not chain compatibility receipts. The new
consumer commit and the original normalization producer must both remain visible
in handoff and run provenance. Old reports are never relabeled or overwritten.
