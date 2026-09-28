# Smart System2: results and token cost

[Figure PNG](success_vs_tokens.png) · [Editable SVG](success_vs_tokens.svg) · [PDF](success_vs_tokens.pdf) · [Exact values CSV](plotted_values.csv) · [Values and source hashes JSON](plotted_values.json)

## Experiment A — Active intervention

- **Language steering improved success while reducing token use relative to TEI.** Astra-guided TLI increased cumulative success from **7/20 (35%) to 13/20 (65%)**, versus **11/20 (55%)** for TEI and **9/20 (45%)** for random-noise retries. TLI used **87.9k Astra tokens per evaluated case**, versus **112.5k** for TEI: **22% fewer tokens** with a ten-percentage-point higher success rate on this cohort. Both allow up to two rescue attempts after baseline failure. TLI plus visual annotations also reached **65%**, at **87.5k tokens/case**; these annotations are separate from the actual pixel perturbations below.
- **Successful TLI rescues usually needed one additional rollout.** Among rescued baseline failures, the median was **one revision and 28.5k tokens** for TLI, versus **two revisions and 121.9k tokens** for TEI. These are success-conditioned medians on different rescued subsets; they exclude cases that never succeeded. The plot instead counts spending on all evaluated cases.
- **Actual image perturbations recovered additional failures.** On a separate twenty-case cohort, Astra-guided occlusion and demonstration-image blending each improved success from **8/20 (40%) to 12/20 (60%)**, using **at least 272.1k** and **281.8k tokens/case**, respectively. Random occlusion reached **50%** and random blending **55%**, with zero Astra tokens. These image arms also allow two rescue attempts.
- **FRS demonstrated successful assisted control in the three-case pilot.** FRS reached **2/3 successes** at **22.2k tokens/case**, compared with **0/3** for both native controls. Direct Astra steering also reached **2/3**, at **at least 29.2k tokens/case**, on a different combination of tasks. This is a small development result, not evidence that FRS is superior or that its efficiency transfers to the twenty-task benchmark.

## Experiment B — System2-in-the-loop improvement

The completed positive results demonstrate **offline reuse and distillation of successful Astra interventions**. They do not establish a repeatedly improving online policy-update loop.

- **Recorded teacher schedules transferred to new resets without online Astra calls.** Schedule replay increased single-attempt success from **15/40 (37.5%) to 27/40 (67.5%)**, recovering **twelve native failures without losing any native successes**, at **0 online Astra tokens/case**.
- **A learned selector retained part of the improvement without online Astra calls.** The observation-conditioned TEI/TLI selector reached **19/40 (47.5%)**, also at **0 online Astra tokens/case**. Its paired outcomes were five recoveries and one lost native success.
- **The small ID panel showed no observed regression.** All five methods succeeded on **8/8 cases across four standard tasks**. The flow head and gated head reached **35% and 40% OOD success**, respectively; the ungated action correction did not improve the native baseline.

Teacher acquisition was not free. The two whole source studies recorded **at least 6,887,020 tokens across 899 calls**, including unsuccessful trials and unused arms. The exact acquisition cost attributable to the twelve selected teachers is unknown. Zero deployment-time Astra tokens therefore demonstrate an online cost saving, not a measured end-to-end amortized advantage. Training and evaluation still required robot-policy compute, reported separately in the original recipe.

The earlier three-task critique/FRS adaptation loop achieved no rescues and no Astra-approved improvements, so it performed **zero optimizer updates**. Its interrupted larger evaluation supplies no completed full-benchmark success rate. This remains separate from the positive offline reuse/distillation results.

## Success versus token cost

![Success versus mean online Astra token cost, separated by study](success_vs_tokens.png)

The horizontal coordinate is **all known online Astra input plus output tokens for an arm divided by all task/reset cases in that cohort**. It includes rejected calls when usage is reported, failed attempts, and zero-intervention baseline successes. It is not the mean over successful rollouts or the mean per provider call. Reasoning tokens are already included in output tokens and are not added twice. The vertical coordinate is simulator success under each panel's protocol. Horizontal scales differ; these are separate descriptive comparisons, not a shared performance frontier.

| Experiment / cohort | Method | Success | Known online tokens, total | Mean tokens / case | Unknown-usage calls |
|---|---|---:|---:|---:|---:|
| A — language, 20 cases | Native baseline | 35% | 0 | 0 | 0 |
| A — language, 20 cases | Random retries | 45% | 0 | 0 | 0 |
| A — language, 20 cases | TEI | 55% | 2,250,708 | 112,535.4 | 0 |
| A — language, 20 cases | TLI | 65% | 1,757,115 | 87,855.8 | 0 |
| A — language, 20 cases | TLI + annotations | 65% | 1,750,151 | 87,507.6 | 0 |
| A — pixels, 20 cases | Native baseline | 40% | 0 | 0 | 0 |
| A — pixels, 20 cases | Random noise | 50% | 0 | 0 | 0 |
| A — pixels, 20 cases | Random occlusion | 50% | 0 | 0 | 0 |
| A — pixels, 20 cases | Random blend | 55% | 0 | 0 | 0 |
| A — pixels, 20 cases | Astra occlusion | 60% | ≥5,441,930 | ≥272,096.5 | 1 |
| A — pixels, 20 cases | Astra blend | 60% | 5,635,176 | 281,758.8 | 0 |
| A — FRS pilot, 3 cases | Native Euler 10 | 0% | 0 | 0 | 0 |
| A — FRS pilot, 3 cases | Native repeated noise | 0% | 0 | 0 | 0 |
| A — FRS pilot, 3 cases | Direct steering | 66.7% | ≥87,680 | ≥29,226.7 | 1 |
| A — FRS pilot, 3 cases | FRS | 66.7% | 66,707 | 22,235.7 | 0 |
| B — offline recipe, 40 cases | Native | 37.5% | 0 | 0 | 0 |
| B — offline recipe, 40 cases | Recorded schedule | 67.5% | 0 | 0 | 0 |
| B — offline recipe, 40 cases | Learned selector | 47.5% | 0 | 0 | 0 |
| B — offline recipe, 40 cases | Flow head | 35% | 0 | 0 | 0 |
| B — offline recipe, 40 cases | Gated flow head | 40% | 0 | 0 | 0 |

Numbers marked **≥** have incomplete provider usage and are lower bounds. Zero Astra tokens does not mean zero VLA compute, simulator compute, human effort, or teacher cost. Token counts include provider-reported multimodal input usage, and are not dollar prices or latency measurements. The plot excludes separate development/interruption overhead for the twenty-case studies, historical teacher acquisition, and uninstrumented coding/review/orchestration tokens. Selected-teacher acquisition cost is unknown, so no amortized cost or break-even deployment count is invented.

These studies use different seeds, reset states, prompts, image inputs, control cadences and retry budgets. Compare methods within each panel. FRS and direct steering here are single-attempt controls on three selected development cases, not the interrupted full FRS benchmark. The complete language/image studies test all twenty published OOD compositions on one reset each; Experiment B uses two new resets per composition. The compositions were known during the research; this is not held-out-task generalization. All measurements use the benchmark's simulator predicates, not a separate human assessment of stable placement.

The figure covers the completed comparisons highlighted in the draft. It does not plot privileged oracle controls, the earlier joint noise/language/annotation search, partial VEI/VLI experiments, the critique/adaptation arms, the interrupted full FRS evaluation or the small ID panel. Their results and costs remain in their original reports.

## Provenance and regeneration

- [Language results and rescue protocol](../phase_interpolation/evaluation/results/report.md)
- [Actual image perturbation results](../image_perturbations/evaluation/README.md)
- [FRS development results](../frs_policy_improvement/development/report.md)
- [Offline reuse/distillation results](../learned_correction_recipe/README.md)
- [Teacher acquisition accounting](../learned_correction_recipe/results/HISTORICAL_CONTEXT.md)

Run `python astra_reversal/reports/smart_system2_results/build_results.py` from the repository's activated Python environment. The builder extracts the values from the original JSON reports, reconciles complete cohort sizes and token subtotals, and exports one four-panel figure as PNG, SVG and PDF. CSV/JSON retain both native FRS controls and both identical-rate random pixel controls even where the plot combines their labels. [manifest.json](manifest.json) binds the source and output bytes. No new policy rollout or inference call is made.
