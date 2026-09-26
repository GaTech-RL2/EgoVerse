# Representation steering — development

Status: **partial**. Seed 19; 1/3 prescribed cases independently audited. Efficacy released: False.

Recorded completed-rollout work: 42 rollouts, 11,849 actions, 24,100 velocity evaluations. Separately, the full recorded ledger contains 220 provider calls and 0 preflight failures, including any unfinished rollout. Shared baseline and numerical probes counted once. Incomplete-run rollout/action/compute totals are lower bounds; in-flight work is not assigned zero cost.

Provider-reported tokens, including rejected calls: input_tokens: 1,113,516 (lower bound; missing usage); output_tokens: 15,530 (lower bound; missing usage); reasoning_tokens: 4,910 (lower bound; missing usage); total_tokens: 1,129,046 (lower bound; missing usage). Reasoning is a subset of output.

Summed rollout wall time: 2741.770s. This is measured work across cases, not elapsed workflow time; all other timings are nested.

Independent archive/provider proofs cover 3 cases: 15,096 arrays and 2,431 recorded generations. They establish **24,700 recorded velocity evaluations**, including 390 setup/probe evaluations once per case. Of these, 24,100 belong to completed rollouts and 600 to unfinished rollouts; these are different scopes, not extra additive totals.

The 2 known unfinished attempts executed **at least 300 additional actions** beyond the exact 11,849 completed-row actions: at least 12,149 actions overall. This uses the latest verified request/generation step. The weaker array-only bound (12,139 overall) overlaps that evidence and is not added again. Outcomes and unsynchronized tails remain unknown; the per-attempt protocol cap does not make missing work complete.

| Case | Unfinished attempt | Additional actions lower bound | Recorded flow evaluations | Outcome |
|---|---|---:|---:|---|
| libero_goal_ood:seed19:task6:state0 | astra_vli_revision2 | 25 | 50 | unknown |
| libero_spatial_ood:seed19:task2:state0 | astra_vli_revision2 | 275 | 550 | unknown |

Success curves and paired comparisons are withheld until the full prescribed cohort and both audits are complete. Recorded costs remain available; unrecorded or in-flight work is unknown.

The recorded provider classified calls as **budget_exceeded**. This is a spending-cap failure, not evidence that the unexecuted Astra operators worked. Available token sums are lower bounds; missing usage and unfinished work remain unknown.

Individually verified completed-rollout observations from preserved archives. This table is not a complete-cohort success rate or a causal steering claim. A forced rollout after a successful baseline is a stability check, not a rescue.

| Case | Actual physical rollout | Recorded outcome | Actions | Accepted / edit-effect / fallback actions | Application scope |
|---|---|---|---:|---|---|
| libero_goal_ood:seed19:task6:state0 | native_revision0 | failure | 300 | 0 / 0 / 0 | native policy |
| libero_goal_ood:seed19:task6:state0 | native_retry_revision1 | failure | 300 | 0 / 0 / 0 | native policy |
| libero_goal_ood:seed19:task6:state0 | native_retry_revision2 | failure | 300 | 0 / 0 / 0 | native policy |
| libero_goal_ood:seed19:task6:state0 | random_tli_revision1 | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_goal_ood:seed19:task6:state0 | random_tli_revision2 | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_goal_ood:seed19:task6:state0 | random_vei_revision1 | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_goal_ood:seed19:task6:state0 | random_vei_revision2 | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_goal_ood:seed19:task6:state0 | random_vli_revision1 | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_goal_ood:seed19:task6:state0 | random_vli_revision2 | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_goal_ood:seed19:task6:state0 | astra_tei_revision1 | failure | 300 | 300 / 300 / 0 | accepted representation choices; native portions may also occur |
| libero_goal_ood:seed19:task6:state0 | astra_tei_revision2 | success | 98 | 98 / 98 / 0 | accepted representation choices; native portions may also occur |
| libero_goal_ood:seed19:task6:state0 | astra_tli_revision1 | failure | 300 | 300 / 300 / 0 | accepted representation choices; native portions may also occur |
| libero_goal_ood:seed19:task6:state0 | astra_tli_revision2 | failure | 300 | 0 / 0 / 300 | native fallback only; no accepted Astra choice executed |
| libero_goal_ood:seed19:task6:state0 | astra_vei_revision1 | failure | 300 | 0 / 0 / 300 | native fallback only; no accepted Astra choice executed |
| libero_goal_ood:seed19:task6:state0 | astra_vei_revision2 | failure | 300 | 0 / 0 / 300 | native fallback only; no accepted Astra choice executed |
| libero_goal_ood:seed19:task6:state0 | astra_vli_revision1 | failure | 300 | 0 / 0 / 300 | native fallback only; no accepted Astra choice executed |
| libero_spatial_ood:seed19:task2:state0 | native_revision0 | failure | 300 | 0 / 0 / 0 | native policy |
| libero_spatial_ood:seed19:task2:state0 | native_retry_revision1 | failure | 300 | 0 / 0 / 0 | native policy |
| libero_spatial_ood:seed19:task2:state0 | native_retry_revision2 | failure | 300 | 0 / 0 / 0 | native policy |
| libero_spatial_ood:seed19:task2:state0 | random_tli_revision1 | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_spatial_ood:seed19:task2:state0 | random_tli_revision2 | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_spatial_ood:seed19:task2:state0 | random_vei_revision1 | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_spatial_ood:seed19:task2:state0 | random_vei_revision2 | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_spatial_ood:seed19:task2:state0 | random_vli_revision1 | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_spatial_ood:seed19:task2:state0 | random_vli_revision2 | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_spatial_ood:seed19:task2:state0 | astra_tei_revision1 | failure | 300 | 300 / 300 / 0 | accepted representation choices; native portions may also occur |
| libero_spatial_ood:seed19:task2:state0 | astra_tei_revision2 | failure | 300 | 300 / 300 / 0 | accepted representation choices; native portions may also occur |
| libero_spatial_ood:seed19:task2:state0 | astra_tli_revision1 | success | 120 | 120 / 120 / 0 | accepted representation choices; native portions may also occur |
| libero_spatial_ood:seed19:task2:state0 | astra_vei_revision1 | failure | 300 | 0 / 0 / 300 | native fallback only; no accepted Astra choice executed |
| libero_spatial_ood:seed19:task2:state0 | astra_vei_revision2 | failure | 300 | 0 / 0 / 300 | native fallback only; no accepted Astra choice executed |
| libero_spatial_ood:seed19:task2:state0 | astra_vli_revision1 | failure | 300 | 0 / 0 / 300 | native fallback only; no accepted Astra choice executed |
| libero_spatial_ood:seed19:task8:state0 | native_revision0 | success | 106 | 0 / 0 / 0 | native policy |
| libero_spatial_ood:seed19:task8:state0 | native_retry_revision1 (forced stability) | failure | 300 | 0 / 0 / 0 | native policy |
| libero_spatial_ood:seed19:task8:state0 | random_tli_revision1 (forced stability) | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_spatial_ood:seed19:task8:state0 | random_vei_revision1 (forced stability) | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_spatial_ood:seed19:task8:state0 | random_vli_revision1 (forced stability) | failure | 300 | 0 / 300 / 0 | random representation choice |
| libero_spatial_ood:seed19:task8:state0 | astra_tei_revision1 (forced stability) | failure | 300 | 300 / 150 / 0 | accepted representation choices; native portions may also occur |
| libero_spatial_ood:seed19:task8:state0 | astra_tli_revision1 (forced stability) | failure | 300 | 300 / 175 / 0 | accepted representation choices; native portions may also occur |
| libero_spatial_ood:seed19:task8:state0 | astra_vei_revision1 (forced stability) | failure | 300 | 300 / 175 / 0 | accepted representation choices; native portions may also occur |
| libero_spatial_ood:seed19:task8:state0 | astra_vli_revision1 (forced stability) | success | 125 | 125 / 50 / 0 | accepted representation choices; native portions may also occur |
| libero_spatial_ood:seed19:task8:state0 | astra_tli_vli_revision1 (forced stability) | failure | 300 | 0 / 0 / 300 | native fallback only; no accepted Astra choice executed |
| libero_spatial_ood:seed19:task8:state0 | astra_pixel_blend_revision1 (forced stability) | failure | 300 | 0 / 0 / 300 | native fallback only; no accepted Astra choice executed |

Forced development stability checks, shown separately. These are the same physical rows listed above, not extra cost or rescues.

| Case | Extra arm | Actual outcome | Actions |
|---|---|---|---:|
| libero_spatial_ood:seed19:task8:state0 | native_retry | failure | 300 |
| libero_spatial_ood:seed19:task8:state0 | random_tli | failure | 300 |
| libero_spatial_ood:seed19:task8:state0 | random_vei | failure | 300 |
| libero_spatial_ood:seed19:task8:state0 | random_vli | failure | 300 |
| libero_spatial_ood:seed19:task8:state0 | astra_tei | failure | 300 |
| libero_spatial_ood:seed19:task8:state0 | astra_tli | failure | 300 |
| libero_spatial_ood:seed19:task8:state0 | astra_vei | failure | 300 |
| libero_spatial_ood:seed19:task8:state0 | astra_vli | success | 125 |
| libero_spatial_ood:seed19:task8:state0 | astra_tli_vli | failure | 300 |
| libero_spatial_ood:seed19:task8:state0 | astra_pixel_blend | failure | 300 |

Recorded operator coverage. Accepted calls during unfinished rollouts are not counted as executed decisions. Zero executed accepted choices means a configured Astra arm did not exercise its requested steering in the completed evidence.

| Astra arm | Recorded calls / accepted | Completed rollouts | Executed accepted decisions | Edit-effect actions | Native fallback actions |
|---|---:|---:|---:|---:|---:|
| astra_tei | 52 / 52 | 5 | 52 | 1148 | 0 |
| astra_tli | 41 / 29 | 4 | 29 | 595 | 300 |
| astra_vei | 60 / 12 | 5 | 12 | 175 | 1200 |
| astra_vli | 43 / 5 | 3 | 5 | 50 | 600 |
| astra_tli_vli | 12 / 0 | 1 | 0 | 0 | 300 |
| astra_pixel_blend | 12 / 0 | 1 | 0 | 0 | 300 |

[Machine report](report.json) · [Case/arm CSV](case_arms.csv) · [Physical rollout CSV](physical_rollouts.csv) · [Verified completed-rollout CSV](verified_completed_rollouts.csv) · [Unfinished-work CSV](unfinished_attempts.csv) · [Budget/token CSV](revision_budgets.csv) · [Forced development CSV](development_stability.csv) · [Protocol](protocol.json)

- Exploratory assisted correction on known task compositions and prescribed resets; no autonomous learning or held-out-task claim.
- Every arm shares one native baseline. Success at revision zero is not an intervention rescue; failures remain in the fixed denominator.
- Revision budgets count full reset rollouts. An online decision every 25 actions is a different unit; both are reported.
- Policy noise is matched across arms at a case/revision/step and changes across revisions. Native retries measure gains available from that new noise alone.
- Random controls use the same source catalogs and continuous alpha bounds. Astra's explicit deferral distribution is not reproduced by random controls.
- Recorded representation changes and accepted proposals are not proofs of action causation. A success can include neutral choices or native fallback after a failed call.
- Physical totals count the shared baseline and numerical probes once. Standalone arm attribution repeats shared setup and must never be summed across arms.
- Rollout wall time includes provider waits, donor captures, probes, recording and simulator work. Policy, environment, conditioning, capture and provider times are nested diagnostics, not extra wall time.
- All physical calls, including rejected calls, contribute available provider usage. Missing usage stays unknown; known sums are lower bounds. Reasoning tokens are a subset of output. No dollar cost is assumed.
- Rescue-only medians exclude baseline successes and censored failures. Failed searches retain their full recorded token and rollout costs.
- Development may force a revision after baseline success. Those extra physical costs are reported separately and do not improve the success curve.
- Partial costs cover recorded work only. In-flight, interrupted, setup and donor-extraction work are not silently assigned zero cost or pooled into the completed cohort.
- Independent audits bind archives, arrays, provider feedback, resets and frozen weights. This report does not independently replay model hidden states or simulator physics.
- Videos contain raw external-camera frames before each executed action, at 20 fps; they omit inference pauses, stabilization and the terminal post-action image.
