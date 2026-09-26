# Astra flow reversal and auxiliary policy improvement — development

Status: **complete**. Recorded tasks: 3/3; passed full task audits: 3.

Retain every prescribed reset; credited success requires recorded success, positive executed actions, and initial_success=false. Recorded flags remain available separately.
Initial-success physical recordings: **0**.

## Same-reset adaptation: cumulative simulator success

| Method | Round / revision | Successes / prescribed episodes |
|---|---:|---:|
| Critique and FRS, no learning | 0 | 0/3 |
| Critique and FRS, no learning | 1 | 0/3 |
| Critique and FRS, no learning | 2 | 0/3 |
| Critique and FRS, no learning | 3 | 0/3 |
| Critique and FRS with auxiliary learning | 0 | 0/3 |
| Critique and FRS with auxiliary learning | 1 | 0/3 |
| Critique and FRS with auxiliary learning | 2 | 0/3 |
| Critique and FRS with auxiliary learning | 3 | 0/3 |

| Method | Rescues / baseline failures | Median revision among rescues | Median tokens among complete-usage rescues | Rescues with missing usage | Censored failures |
|---|---:|---:|---:|---:|---:|
| Critique and FRS, no learning | 0/3 | None | None | 0 | 3 |
| Critique and FRS with auxiliary learning | 0/3 | None | None | 0 | 3 |

| Final-round task | Critique and FRS, no learning | Critique and FRS with auxiliary learning |
|---|---:|---:|
| libero_goal_ood:6: put the wine bottle in the bowl | 0/1 | 0/1 |
| libero_spatial_ood:2: put the milk on the plate | 0/1 | 0/1 |
| libero_spatial_ood:8: put the bowl at table center on the cabinet | 0/1 | 0/1 |

| Suite | Method | Round / revision | Successes / episodes |
|---|---|---:|---:|
| libero_goal_ood | Critique and FRS, no learning | 0 | 0/1 |
| libero_spatial_ood | Critique and FRS, no learning | 0 | 0/2 |
| libero_goal_ood | Critique and FRS, no learning | 1 | 0/1 |
| libero_spatial_ood | Critique and FRS, no learning | 1 | 0/2 |
| libero_goal_ood | Critique and FRS, no learning | 2 | 0/1 |
| libero_spatial_ood | Critique and FRS, no learning | 2 | 0/2 |
| libero_goal_ood | Critique and FRS, no learning | 3 | 0/1 |
| libero_spatial_ood | Critique and FRS, no learning | 3 | 0/2 |
| libero_goal_ood | Critique and FRS with auxiliary learning | 0 | 0/1 |
| libero_spatial_ood | Critique and FRS with auxiliary learning | 0 | 0/2 |
| libero_goal_ood | Critique and FRS with auxiliary learning | 1 | 0/1 |
| libero_spatial_ood | Critique and FRS with auxiliary learning | 1 | 0/2 |
| libero_goal_ood | Critique and FRS with auxiliary learning | 2 | 0/1 |
| libero_spatial_ood | Critique and FRS with auxiliary learning | 2 | 0/2 |
| libero_goal_ood | Critique and FRS with auxiliary learning | 3 | 0/1 |
| libero_spatial_ood | Critique and FRS with auxiliary learning | 3 | 0/2 |

## Separate-reset policy evaluation (development only)

| Method | Round / revision | Successes / prescribed episodes |
|---|---:|---:|
| Native Euler 10 | 3 | 0/3 |
| Native repeated noise | 3 | 0/3 |
| Astra direction, direct execution | 3 | 2/3 |
| Astra flow reversal steering | 3 | 2/3 |
| Critique and FRS, no learning | 3 | 0/3 |
| Learned auxiliary noise policy | 1 | 0/3 |
| Learned auxiliary noise policy | 2 | 0/3 |
| Learned auxiliary noise policy | 3 | 0/3 |

Checkpoint success is not accumulated across rounds. Static comparators have one final-round measurement; no unexecuted earlier measurement is invented.

| Final-round task | Native Euler 10 | Native repeated noise | Astra direction, direct execution | Astra flow reversal steering | Critique and FRS, no learning | Learned auxiliary noise policy |
|---|---:|---:|---:|---:|---:|---:|
| libero_goal_ood:6: put the wine bottle in the bowl | 0/1 | 0/1 | 1/1 | 1/1 | 0/1 | 0/1 |
| libero_spatial_ood:2: put the milk on the plate | 0/1 | 0/1 | 0/1 | 1/1 | 0/1 | 0/1 |
| libero_spatial_ood:8: put the bowl at table center on the cabinet | 0/1 | 0/1 | 1/1 | 0/1 | 0/1 | 0/1 |

| Suite | Method | Round / revision | Successes / episodes |
|---|---|---:|---:|
| libero_goal_ood | Native Euler 10 | 3 | 0/1 |
| libero_spatial_ood | Native Euler 10 | 3 | 0/2 |
| libero_goal_ood | Native repeated noise | 3 | 0/1 |
| libero_spatial_ood | Native repeated noise | 3 | 0/2 |
| libero_goal_ood | Astra direction, direct execution | 3 | 1/1 |
| libero_spatial_ood | Astra direction, direct execution | 3 | 1/2 |
| libero_goal_ood | Astra flow reversal steering | 3 | 1/1 |
| libero_spatial_ood | Astra flow reversal steering | 3 | 1/2 |
| libero_goal_ood | Critique and FRS, no learning | 3 | 0/1 |
| libero_spatial_ood | Critique and FRS, no learning | 3 | 0/2 |
| libero_goal_ood | Learned auxiliary noise policy | 1 | 0/1 |
| libero_spatial_ood | Learned auxiliary noise policy | 1 | 0/2 |
| libero_goal_ood | Learned auxiliary noise policy | 2 | 0/1 |
| libero_spatial_ood | Learned auxiliary noise policy | 2 | 0/2 |
| libero_goal_ood | Learned auxiliary noise policy | 3 | 0/1 |
| libero_spatial_ood | Learned auxiliary noise policy | 3 | 0/2 |

## Recorded physical cost

45 unique rollouts; 12,766 actions; 14,720 velocity evaluations; 0 auxiliary optimizer steps.

775 physical provider calls: 771 accepted, 4 failed/rejected; 0 additional no-network preflight failures.

Input tokens: 3,060,656 known (3 missing); output: 169,048 known (3 missing); total: **3,229,704 known (3 missing)**; reasoning subset: 23,128 known (3 missing).

All recorded physical provider calls, including failures and preflights, are partitioned once. Online-call latency and auxiliary inference are already inside rollout wall time. Critique/judge latency and training time are additional. Worker bootstrap, checkpoint I/O and unrecorded/in-flight work are not inferred.

## Recorded auxiliary updates

| Task | Round | Astra promoted | Optimizer steps | Replay samples | Actor state |
|---|---:|---|---:|---:|---|
| libero_goal_ood:6 | 1 | False | 0 | 0 | `1d7cc2cf4280ba63e69ea3012253a3edd276b95ac476f71284e10e31583f5dff` |
| libero_goal_ood:6 | 2 | False | 0 | 0 | `1d7cc2cf4280ba63e69ea3012253a3edd276b95ac476f71284e10e31583f5dff` |
| libero_goal_ood:6 | 3 | False | 0 | 0 | `1d7cc2cf4280ba63e69ea3012253a3edd276b95ac476f71284e10e31583f5dff` |
| libero_spatial_ood:2 | 1 | False | 0 | 0 | `2f65d73bf45f9dbb0dc1f6a1f1a0702bdb4dbe8e29094fea5d0a124af40680ff` |
| libero_spatial_ood:2 | 2 | False | 0 | 0 | `2f65d73bf45f9dbb0dc1f6a1f1a0702bdb4dbe8e29094fea5d0a124af40680ff` |
| libero_spatial_ood:2 | 3 | False | 0 | 0 | `2f65d73bf45f9dbb0dc1f6a1f1a0702bdb4dbe8e29094fea5d0a124af40680ff` |
| libero_spatial_ood:8 | 1 | False | 0 | 0 | `b747eae4367bd8630d20cf7878488dbc83bd951d8abc80ed07ba96b43c7361a9` |
| libero_spatial_ood:8 | 2 | False | 0 | 0 | `b747eae4367bd8630d20cf7878488dbc83bd951d8abc80ed07ba96b43c7361a9` |
| libero_spatial_ood:8 | 3 | False | 0 | 0 | `b747eae4367bd8630d20cf7878488dbc83bd951d8abc80ed07ba96b43c7361a9` |

## Interpretation

All tasks are known published OOD compositions. Separate reset indexes are not held-out tasks; checkpoint training overlap is unknown.

Development and evaluation are separate. Development evaluates only reset1 of its three selected tasks; evaluation uses resets1–10 of all20 tasks.

Adaptation always runs three revisions on reset0. Cumulative retry success does not measure a deployed learned policy. Checkpoint evaluations do not return feedback or labels to adaptation.

Astra judgments, simulator predicates and auxiliary update acceptance are separate. Judges receive observable trajectories, not success labels; early environment stopping can reveal trajectory length.

Only the auxiliary noise actor is trained. Native VLA tensor bytes are bound before and after each fully audited worker or sealed completed task. Online steering methods and the learned-noise evaluation have different provider/computation costs.

Retain every prescribed reset; credited success requires recorded success, positive executed actions, and initial_success=false. Recorded flags remain available separately. Recorded initial-success physical runs: 0.

Shared adaptation baseline recordings count once in physical totals. Static evaluation controls are executed once at the final round, not duplicated physically at earlier checkpoints.

Physical costs cover the supplied authoritative whole-task records. Aborted or replaced task attempts require separate cost reports and are not silently pooled into this cohort.

Tokens through first-success revision include that completed revision's critique, online actions and judge, including failed calls. Later fixed revisions remain in the full physical cost.

All recorded physical provider calls, including failures and preflights, are partitioned once. Online-call latency and auxiliary inference are already inside rollout wall time. Critique/judge latency and training time are additional. Worker bootstrap, checkpoint I/O and unrecorded/in-flight work are not inferred.

No dollar price or superiority over matched RL efficiency is asserted. The renderer and producer do not rerun hidden model computation or simulator physics.

Exact system prompts, hash-bound source inventories, per-call token availability and per-round training/checkpoint details are retained in report.json. No new experiment, model, provider or simulator call was made to generate this report.

## Approach and downloads

The approach is a hash-bound narrative snapshot; the recorded protocol and exact worker prompt manifest remain authoritative. Audit downloads use canonical JSON serialization, with their original file hashes and encoding retained in the audit index.

- [Complete FRS report JSON](report.json)
- [Detailed Markdown report](report.md)
- [Episode / method / round CSV](episodes.csv)
- [Per-suite outcomes, conditional rescues and paired comparisons](supporting_results.json)
- [Exact approach narrative](approach/index.html)
- [Unchanged APPROACH.md snapshot](approach/APPROACH.md)
- [All four exact worker prompts](prompts.json)
- [Recorded protocol](protocol.json)
- [Passed independent task audits](audits/index.html)
- [Audit source/download hash index](audits/index.json)
- [Adaptation and checkpoint figure (PNG)](figures/success_and_learning.png)
- [Adaptation and checkpoint figure (PDF)](figures/success_and_learning.pdf)
- [Adaptation and checkpoint figure (SVG)](figures/success_and_learning.svg)
- [Unique physical token cost figure (PNG)](figures/provider_token_cost.png)
- [Unique physical token cost figure (PDF)](figures/provider_token_cost.pdf)
- [Unique physical token cost figure (SVG)](figures/provider_token_cost.svg)
- [Exact plotted values and source hashes](figures/plotted_values.json)
- [Figure file hashes and Matplotlib version](figures/manifest.json)
- [Exact postprocessor source: frs_report.py](source/frs_report.py.txt)
- [Exact postprocessor source: frs_html_report.py](source/frs_html_report.py.txt)
- [Exact postprocessor source: frs_plots.py](source/frs_plots.py.txt)
- [Complete bundle file hashes](manifest.json)

## Related studies — separate cohorts

These earlier measurements are context only. Their outcomes, calls, tokens and interrupted-run overhead are excluded from every FRS table, curve and physical total above.

### Historical 200-episode OOD baseline

Native Euler 10: 86/200 successes after whole-shard hardware-error repair; 0 canonical execution errors.

Seed 7, ten reset-stream trials per known task, five actions per policy call, TF32 enabled. FRS uses different resets, a ten-action execution chunk and TF32 disabled. This is historical context, not a matched FRS comparator; original hardware errors and repair overhead remain in its own report.

[Study and evidence](related/baseline/report.json) · [Exact result JSON](related/baseline/report.json) · SHA256 `40c6828cc18cf166663d88d33f656da7bda26215d9c1eed341522282cb72b0b9`.

### Completed image perturbation study

Seed 37, one reset per each of 20 known tasks, up to two revisions. Common baseline 8/20; Astra occlusion 12/20 and donor-image blending 12/20; matched random occlusion 10/20 and blending 11/20. These are cumulative retry outcomes.

Actual RGB perturbations with a different decision schedule, five-action execution chunks, reset budget and feedback contract. These outcomes are not learned-policy checkpoint measurements. Its failed-call, fallback, missing-usage and interruption costs remain separate; none is added to FRS totals.

[Study and evidence](related/image_study/index.html) · [Exact result JSON](related/image_study/results/report.json) · SHA256 `bd2ae73794dcdf9c73f9033c8934ab23ce8ea2ab0b878b239d2af8c963adff87`.
