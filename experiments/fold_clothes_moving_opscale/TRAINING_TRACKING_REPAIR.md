# Raw tracking validity repair

## Why a fresh attempt is required

Missing hand tracking is represented in the raw numeric arrays by a wrist-pose
sentinel and an all-zero keypoint block. The values are finite. Earlier training
could therefore accept them as real targets; its outer normalization bounds
also included missing values. Gradient clipping limits update magnitude but
does not reject such targets. The median/MAD spike detector is diagnostic only.
The pinned reference recipe disables weight EMA, and has no EMA loss-rejection
guard. Weight averaging would not make missing targets valid.

The original two-update smoke phases did not sample enough real training to
expose this intermittent failure. Affected optimizer checkpoints must not be
resumed. New production identities start from fresh model and optimizer state.

## Corrected sampling

The opt-in `raw_tracking_valid_windows_v1` policy checks raw tracking for the
current frame, real history, and the full future interpolation support before
numeric transforms or image decoding. An invalid requested start is replaced
with a valid start in the same episode. Decoder retries follow the same rule.
The original episode list and index-map weights remain intact; batch size is64.
An episode with zero valid starts is a hard failure and is never silently dropped.
Nominal selected source hours remain the experimental budget; reports separately
count invalid frames and rejected windows.

Normalization adopts the full training-history eligibility even though its
numeric statistics view uses one history frame. Guarded normalization bypasses
the old per-episode strided-row cache and draws the same maximum200,000 rows
through the guarded DataLoader. This intentionally changes which rows can enter
statistics. The final content cache remains enabled, bound to the full training
configuration, policy, configured seed, source, and episode fingerprints. The
configured seed42026 is now passed explicitly; earlier precompute calls could
use the local default42. Python/NumPy sampling state is scoped and restored for
direct zero-worker calls. Old normalization artifacts are not reused.

## Required evidence before production

1. Fresh data readiness and all36 policy-bound normalization reports, with exact
   selected-episode coverage and positive valid-start counts for every episode.
2. A real op16 diagnostic of512 full training updates, including a forced invalid
   raw start that is replaced within the same episode without advancing training
   random-number state.
3. The existing op4 evaluation/resume and op128 evaluation phases with64 updates
   each; op4 resumes all training state to128 total updates.
4. All actual batches are64 and use valid raw windows. Every diagnostic update
   records finite loss and gradient norm. Loss greater than1000 fails preflight;
   this diagnostic threshold never clips or modifies the training objective.
5. Existing validation coverage,800-batch cohort limits, videos, remote W&B,
   checkpoint checks, and reopened artifact hashes must all pass.

The RDT architecture, optimizer/schedule,400,000-step target, batch sizes,
operator/split design, and weight-EMA setting stay those of the pinned recipe.
