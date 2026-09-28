# Full FRS evaluation and VEI/VLI screen

**Status: prepared; GPU evaluations have not launched.** A fresh OSMO provider
check on September 28, 2026 returned HTTP 429 with structured error type
`budget_exceeded`. The existing inference credential must be funded before
either study can execute. This is not a benchmark result. The check used one
small provider request and no GPU. Its [receipt](provider_check.json) and
[OSMO workflow](https://us-west-2-aws.osmo.nvidia.com/workflows/astra-pi05-provider-check-20260928-1)
record the failure without exposing credentials or raw provider response text.

## Prescribed comparisons

| Study | Task/reset cases | Methods | Attempts | Maximum physical rollouts | Maximum Astra calls |
|---|---:|---|---|---:|---:|
| Full FRS | 20 tasks × 10 resets = 200 | Native Euler-10, native repeated noise, Astra FRS | One per method/case | 600 | 6,000 |
| VEI/VLI screen | 20 tasks × 1 reset = 20 | Shared native baseline; native retry, random VEI, random VLI, Astra VEI, Astra VLI | Baseline, then up to two revisions per arm after failure | 220 | 960 |

Both studies cover all ten LIBERO-Goal-OOD and all ten LIBERO-Spatial-OOD
compositions. FRS uses seed 43 and reset IDs 1–10. The visual screen uses the
existing prescribed seed 47/reset 0. These are fresh campaigns; the old
interrupted records will not be pooled into their success denominators.
The user has been asked whether to expand the visual screen to ten resets;
the prepared version retains its existing one-reset scope pending that choice.

The checkpoint remains `lerobot/pi05_libero_base`, revision
`a217bfd3b14673cf2ce597e69997ab21866438dd`, with the verified OpenPI LIBERO input
profile and quantile normalization. Each worker checks native numerical parity
and strictly loaded weights on its allocated L40S before evaluating. Frozen
parameter hashes and reset provenance are retained per completed task.

FRS uses the existing Astra direction prompt, ten-action execution/query cadence
and ten-step Euler solver. Accepted directional references are inverted through
the frozen policy and regenerated, with the existing padding-noise handling.
This comparison has no direct-steering arm, critique adaptation, auxiliary actor
or policy updates. Native fresh-noise and repeated-noise controls both remain.

VEI interpolates current post-projector visual tokens toward one paired standard
demonstration frame. VLI interpolates the corresponding visual slots after
blocks 0–16. Donor representations use the current target instruction, with raw
current proprioception retained. Astra receives current raw camera observations,
donor previews, robot state and its own preceding rollout feedback. It chooses
native conditioning or a donor and coefficient every 25 actions; the policy
replans every five actions. Native retries and random VEI/VLI use the same reset
and policy-noise streams at corresponding revision/action indexes.

Both use Astra at medium reasoning effort with the existing 8,192 completion
token cap and no hidden retries. A 300-action rollout ends at simulator success
or the action cap. FRS reports single-attempt SR over 200 cases. The screen
reports native SR, cumulative case SR by revision, conditional rescues, and
native-retry/random controls; all twenty cases stay in each denominator.

## Provider interruption and evidence

The new protocols stop on the first observed provider budget, authorization,
rate-limit or request-preflight failure in a worker. The failed call remains in
the ledger. The current rollout remains unfinished rather than being labeled a
native-fallback success or failure; final worker archives retain the available
prefix. Ordinary rejected proposals and isolated HTTP 503 responses keep the
declared per-call fallback behavior and remain visible in the results.
No worker automatically retries a blocked call or relaunches itself.

Report iterations and actions to success, censored failures, all reported input
plus output tokens, missing-usage calls, decision latency, whole-rollout time,
policy/simulator time and inversion/forward evaluations. The videos retain raw
observations at 20 fps; their playback excludes inference pauses. Report partial
coverage and costs if interrupted; release full-cohort SR only after all
prescribed cases and their archives are verified.

## Reproducible run definitions

- [FRS protocol](../../configs/frs_frozen_evaluation_v1.json)
- [FRS OSMO specification](../../osmo/frs_frozen_evaluation_l40s.yaml)
- [VEI/VLI protocol](../../configs/vision_representation_screen_v1.json)
- [VEI/VLI OSMO specification](../../osmo/vision_representation_screen_l40s.yaml)
- [Provider stop behavior](../../provider_stop.py)

The OSMO specifications request eight L40S workers per study, using the existing
bootstrap, checkpoint loader, donor library and archive mechanism. They have
passed OSMO schema dry-runs. A launch still requires a checksummed payload from
the committed revision, a successful provider availability check, and recording
the actual workflow identity. No full-study workflow has been submitted.

Local verification: 1,424 unit tests passed and five optional OpenPI tests were
skipped. Synthetic rollout tests cover all ten FRS evaluation resets, absence
of adaptation, the visual arm subset, and preserving failed-call evidence while
stopping before fallback actions. Ruff checks passed. These checks do not
substitute for the GPU integration gates or task-success measurements.
