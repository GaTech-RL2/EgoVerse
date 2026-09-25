# Phase interpolation: evaluation

Complete audited coverage: **20 fixed cases**, seed 29. Common recovered-noise baseline: **7/20**.

Physical case execution: 190 rollouts, 46,653 actions, 119,140 velocity evaluations. Provider: 679 physical calls, 26 failed calls, 0 preflight failures.
Physical reported tokens: input **5,543,789**, output **214,185**, total **5,757,974**; reasoning subset **24,960**. Dollar cost unknown.

A budget is **up to** that many full rollouts including the common baseline. Oracle caps at 2; random/Astra cap at 3. Median revisions and tokens below are conditional on rescue, so baseline successes do not dominate them. Token medians use rescues with complete reported usage; missing-usage counts remain in the JSON/CSV.

## pooled (20 cases; 13 baseline failures)

| Arm | Cap | Success | Rescues | Censored | Median rescue revisions | Median rescue decisions | Median rescue tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_noise | 3 | 9/20 | 2/13 | 11 | 1.5 | 0.0 | 0.0 |
| oracle_tei | 2 | 10/20 | 3/13 | 10 | 1 | 0 | 0 |
| oracle_tli | 2 | 12/20 | 5/13 | 8 | 1 | 0 | 0 |
| oracle_tei_tli | 2 | 13/20 | 6/13 | 7 | 1.0 | 0.0 | 0.0 |
| astra_tei | 3 | 11/20 | 4/13 | 9 | 2.0 | 16.5 | 121924.0 |
| astra_tli | 3 | 13/20 | 6/13 | 7 | 1.0 | 5.0 | 28529.5 |
| astra_tli_vision | 3 | 13/20 | 6/13 | 7 | 1.0 | 5.0 | 28777.5 |

Native controls: recovered_noise 7/20; known_noise 7/20; policy_fresh 8/20.

## libero_goal_ood (10 cases; 7 baseline failures)

| Arm | Cap | Success | Rescues | Censored | Median rescue revisions | Median rescue decisions | Median rescue tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_noise | 3 | 5/10 | 2/7 | 5 | 1.5 | 0.0 | 0.0 |
| oracle_tei | 2 | 6/10 | 3/7 | 4 | 1 | 0 | 0 |
| oracle_tli | 2 | 6/10 | 3/7 | 4 | 1 | 0 | 0 |
| oracle_tei_tli | 2 | 7/10 | 4/7 | 3 | 1.0 | 0.0 | 0.0 |
| astra_tei | 3 | 4/10 | 1/7 | 6 | 1 | 4 | 21553 |
| astra_tli | 3 | 7/10 | 4/7 | 3 | 1.0 | 5.5 | 31157.0 |
| astra_tli_vision | 3 | 7/10 | 4/7 | 3 | 1.0 | 5.5 | 31757.0 |

Native controls: recovered_noise 3/10; known_noise 3/10; policy_fresh 4/10.

## libero_spatial_ood (10 cases; 6 baseline failures)

| Arm | Cap | Success | Rescues | Censored | Median rescue revisions | Median rescue decisions | Median rescue tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_noise | 3 | 4/10 | 0/6 | 6 | — | — | — |
| oracle_tei | 2 | 4/10 | 0/6 | 6 | — | — | — |
| oracle_tli | 2 | 6/10 | 2/6 | 4 | 1.0 | 0.0 | 0.0 |
| oracle_tei_tli | 2 | 6/10 | 2/6 | 4 | 1.0 | 0.0 | 0.0 |
| astra_tei | 3 | 7/10 | 3/6 | 3 | 2 | 17 | 126787 |
| astra_tli | 3 | 6/10 | 2/6 | 4 | 1.0 | 5.0 | 28529.5 |
| astra_tli_vision | 3 | 6/10 | 2/6 | 4 | 1.0 | 5.0 | 28777.5 |

Native controls: recovered_noise 4/10; known_noise 4/10; policy_fresh 4/10.

## Interpretation limits

- Exploratory known-task follow-up: evaluation uses 20 previously observed OOD compositions with new seed 29 resets; development uses 3 previously inspected seed 19 cases. This is not a held-out-task or zero-shot generalization claim.
- Oracle source pairs/schedules use privileged manual mappings; Astra receives only the restricted nine-donor prompt library, raw observations, robot proprioception and its own feedback.
- This is an explicit PI05 paper-form port, including instruction-only token alignment and all-timestep demonstration means; it is not a reproduction of the released PI0 implementation.
- Budgets mean up to full rollouts including a common baseline: oracle stops at 2; random/Astra stop at 3. Early success reduces actual rollouts. Within-rollout decisions are a separate count.
- Baseline successes cost zero intervention calls. Rescue-only medians condition on baseline failure followed by a later success; censored cases remain in all-case success denominators.
- Provider-reported reasoning tokens are a subset of output tokens. Missing usage remains unknown; partial sums are not complete cost. Preflight failures consume decision slots but make no provider call. No dollar price is assumed.
- Physical case totals count the shared baseline and initialization/gates once. Standalone arm attribution repeats that common setup; its totals must not be summed across arms.
- One-time donor extraction is separate from case compute. Velocity evaluations count action-flow calls, not prefix-encoder forwards. Rollout wall times include baseline initialization/gates, recording and synchronization.
- This audit verifies frozen manifests, recorded execution, provider bindings and feedback image hashes. It does not replay simulator physics, recompute native hidden states, or verify every referenced NPY tensor.
- Development forces an extra rollout per arm even after baseline success. Those physical calls/actions are reported separately and do not establish intervention efficacy.
