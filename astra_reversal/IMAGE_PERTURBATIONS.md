# Image input perturbation study

This study changes actual RGB policy inputs using standard training demonstration
frames and neutral occlusion. The older phase-interpolation experiment used
magenta annotations; its results do not measure these operators.

The frozen protocol is `configs/image_perturbations_v1.json`. Run three previously
inspected seed 19 development cases first, then 20 known OOD compositions at new
seed 37 resets. Baselines are rerun on these resets; comparisons with prior seed 29
results are unpaired. This is exploratory adaptation, with unknown checkpoint
training overlap and a restricted training donor library.

## Operators

For each camera, Astra may select one of:

- Demo blending: `edited=(1-alpha)*raw+alpha*donor`, with alpha from 0 to 1.
  Alpha 1 replaces the whole camera frame. Donors use the same camera, and come
  from five evenly spaced frames of the first pinned standard training episode
  for each of nine donor tasks (45 paired frames). No OOD demonstrations are used.
- Occlusion: blend a constant `[127,127,127]` fill into an integer half-open
  rectangle occupying at most 50% of the camera. Strength ranges from 0 to 1;
  strength 1 fully hides that region. Outside pixels remain byte-identical.

Both use float64 separate convex terms, clipping and round-half-up to uint8.
The library uses official OpenPI resize-with-padding to 224 pixels and retains
the upstream image orientation. Operators do not alter geometry, robot state,
the target instruction, or the fixed recovered noise. Empty operations, alpha 0
and strength 0 are exact no-ops.

## Comparison and feedback

Each case has a common recovered-noise baseline, a known-noise control and a
native fresh-noise control. Five arms get at most two revisions following the
common baseline: random noise, random occlusion, random demo blending, Astra
occlusion and Astra demo blending. Stop each arm after its first success.
Development forces at least one intervention for wiring validation.

Astra sees up to four recent raw paired observations, the previous rollout's
raw snapshots, its own decisions and outcome, and two contact sheets labeling
all 45 paired demo samples. It receives no simulator object poses or dense reward.
It chooses a new image specification every 25 environment actions (at most 12 calls
per 300-action rollout). The same specification is applied independently to each
fresh observation at the five policy replans in that interval. At refresh the
old specification expires; rejected calls clear it and use raw inputs until the
next scheduled decision. There are no hidden retries or inferred replacement
choices. Model choices are bound using a short request ID plus a locally attached
full immutable fingerprint.

Random image arms use the same operator bounds and decision frequency. A
deterministic independent RNG per case, mode and revision selects a 10% empty
choice; otherwise external camera, wrist camera or both with equal probability.
Demo ID is uniform over 45, alpha uniform on [0,1). Occlusion width is uniform over
integer 1..W, height over 1..min(H,floor(0.5*W*H/width)), and origin over valid
positions; strength is uniform on [0,1). This samples the legal action space,
without claiming a uniform distribution over all possible pixel edits.

## Validation and reporting

Each real-checkpoint case first passes the existing flow inversion and native
action parity gates. Additional image gates use the same known noise to require
exact actions for three no-op variants and nonzero controlled-channel and decoded
action changes for fixed nonzero blend and occlusion probes. These probes do not
execute environment actions. Full raw and edited RGB arrays, donor hashes,
specifications, pixel metrics, conditioning IDs, latent arrays and generated and
executed action arrays are retained for independent replay auditing.

Report success after each revision, revisions and online decisions to first
success, and all provider input/output/reasoning usage, including rejected calls.
Count common physical baselines once, while preserving each arm's standalone
cost through success or the cap. Dollar costs require actual provider pricing.
This study measures image effects with fixed language/noise; combined image,
language and noise optimization requires a separate comparison.
