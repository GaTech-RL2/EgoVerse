# Phase interpolation: development

Complete audited coverage: **3 fixed cases**, seed 19. Common recovered-noise baseline: **0/3**.

Physical case execution: 37 rollouts, 8,702 actions, 21,290 velocity evaluations. Provider: 105 physical calls, 3 failed calls, 0 preflight failures.
Physical reported tokens: input **767,744**, output **31,349**, total **799,093**; reasoning subset **3,451**. Dollar cost unknown.

A budget is **up to** that many full rollouts including the common baseline. Oracle caps at 2; random/Astra cap at 3. Median revisions and tokens below are conditional on rescue, so baseline successes do not dominate them. Token medians use rescues with complete reported usage; missing-usage counts remain in the JSON/CSV.

## pooled (3 cases; 3 baseline failures)

| Arm | Cap | Success | Rescues | Censored | Median rescue revisions | Median rescue decisions | Median rescue tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_noise | 3 | 0/3 | 0/3 | 3 | — | — | — |
| oracle_tei | 2 | 1/3 | 1/3 | 2 | 1 | 0 | 0 |
| oracle_tli | 2 | 2/3 | 2/3 | 1 | 1.0 | 0.0 | 0.0 |
| oracle_tei_tli | 2 | 2/3 | 2/3 | 1 | 1.0 | 0.0 | 0.0 |
| astra_tei | 3 | 1/3 | 1/3 | 2 | 1 | 4 | 21668 |
| astra_tli | 3 | 3/3 | 3/3 | 0 | 1 | 5 | 28421 |
| astra_tli_vision | 3 | 3/3 | 3/3 | 0 | 1 | 5 | 29281 |

Native controls: recovered_noise 0/3; known_noise 0/3; policy_fresh 0/3.

## libero_goal_ood (1 cases; 1 baseline failures)

| Arm | Cap | Success | Rescues | Censored | Median rescue revisions | Median rescue decisions | Median rescue tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_noise | 3 | 0/1 | 0/1 | 1 | — | — | — |
| oracle_tei | 2 | 1/1 | 1/1 | 0 | 1 | 0 | 0 |
| oracle_tli | 2 | 1/1 | 1/1 | 0 | 1 | 0 | 0 |
| oracle_tei_tli | 2 | 1/1 | 1/1 | 0 | 1 | 0 | 0 |
| astra_tei | 3 | 0/1 | 0/1 | 1 | — | — | — |
| astra_tli | 3 | 1/1 | 1/1 | 0 | 2 | 17 | 132062 |
| astra_tli_vision | 3 | 1/1 | 1/1 | 0 | 2 | 17 | 134638 |

Native controls: recovered_noise 0/1; known_noise 0/1; policy_fresh 0/1.

## libero_spatial_ood (2 cases; 2 baseline failures)

| Arm | Cap | Success | Rescues | Censored | Median rescue revisions | Median rescue decisions | Median rescue tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_noise | 3 | 0/2 | 0/2 | 2 | — | — | — |
| oracle_tei | 2 | 0/2 | 0/2 | 2 | — | — | — |
| oracle_tli | 2 | 1/2 | 1/2 | 1 | 1 | 0 | 0 |
| oracle_tei_tli | 2 | 1/2 | 1/2 | 1 | 1 | 0 | 0 |
| astra_tei | 3 | 1/2 | 1/2 | 1 | 1 | 4 | 21668 |
| astra_tli | 3 | 2/2 | 2/2 | 0 | 1.0 | 4.5 | 25175.0 |
| astra_tli_vision | 3 | 2/2 | 2/2 | 0 | 1.0 | 5.0 | 29179.5 |

Native controls: recovered_noise 0/2; known_noise 0/2; policy_fresh 0/2.

Development forced extras after first success: **0 rollouts**. Their costs are included in physical totals and excluded from success-prefix token curves. Passing integration demonstrates executed proposals and measured hooks, not intervention efficacy.

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
