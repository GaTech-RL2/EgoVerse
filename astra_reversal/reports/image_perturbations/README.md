# Actual image perturbations

This study changes RGB pixels before the frozen pi0.5 policy sees them: actual
same-camera demo-frame blending, including full replacement, or filled neutral
occlusion. Language and recovered noise stay fixed. Astra sees unmodified
rollout observations and revises its edits every 25 environment actions.
See the [frozen protocol](../../IMAGE_PERTURBATIONS.md) and
[donor manifest](../../configs/image_donors_v1.json).

The audited evaluation covers all 20 OOD tasks at reset 0 and seed 37, with at
most two revisions after the shared baseline. The recovered-noise baseline
succeeds on 8/20 cases; random noise and random occlusion each reach 10/20,
random demo blending reaches 11/20, and Astra occlusion and Astra demo blending
each reach 12/20. These are exploratory results from one reset per task.

The [detailed evaluation report](evaluation/README.md) and
[standalone HTML report](evaluation/index.html) include every task, actual pixel
edits, diagrams, iterations and costs. The canonical evaluation records 440
provider calls and at least 11,077,106 tokens; one HTTP 503 has unknown usage.
Interrupted and development costs are reported separately. The
[independent reconciliation](evaluation/independent_reconciliation.json)
confirms the task inventory, outcomes, provider accounting and published images.

## Audited development

These three previously inspected seed 19 cases validate execution; they are not
an estimate of held-out generalization. All three native controls score 0/3.
Each intervention has at most two revisions after the shared failed baseline.

| Arm | Rescued cases | Revisions for rescue | Astra calls through rescue | Tokens through rescue |
|---|---:|---:|---:|---:|
| Random noise | 0/3 | censored | 0 | 0 |
| Random occlusion | 1/3 (cabinet) | 2 | 0 | 0 |
| Random demo blend | 1/3 (cabinet) | 1 | 0 | 0 |
| Astra occlusion | 1/3 (milk) | 1 | 4 | 87,225 |
| Astra demo blend | 1/3 (cabinet) | 1 | 5 | 112,114 |

Occlusion hides the foreground bowl in the milk task while preserving the raw
wrist camera. That run succeeds in 98 actions; the failed baseline executes 300.
The demo-blending arm does not rescue milk within its two-revision cap. These
[actual input images](development/milk_example/astra_occlusion/comparison.png)
and [demo blend comparison](development/milk_example/astra_demo_blend/comparison.png)
show the first pixel-changing choice in each arm for this illustrative case,
with labels outside the recorded pixels. [Bindings](development/milk_example/examples.json)
retain exact parameters, donor/array hashes and provider decisions.

The cabinet task is also rescued by both random image controls. Astra therefore
has no aggregate advantage over the matched random image controls in these three
cases. Wine fails in every arm. These observations are not a 100% success recipe.

Total development work is 36 physical rollouts, 9,998 environment actions and
23,880 velocity evaluations. All 105 Astra calls are accepted and all usage is
present: **2,647,564 tokens** (2,616,623 input + 30,941 output; 4,919 reasoning tokens
are a subset of output). There are no provider fallbacks. These totals include
failed searches; the table's token columns count only through each rescue.
No verified monetary price is available.

The [complete report](development/results/README.md),
[case/arm table](development/results/case_arms.csv),
[physical-attempt table](development/results/physical_attempts.csv), and
[structured results](development/results/report.json) preserve censored cases,
per-arm costs and paired outcomes. All 18,609 recorded arrays and 2,001 policy
generations pass the [independent worker audits](development/audits/worker_0_receipt.json):
[worker1](development/audits/worker_1_receipt.json),
[worker2](development/audits/worker_2_receipt.json). Pixel replay is exact;
simulator physics and hidden model activations are not independently rerun.

[Provenance](provenance/development.json) records the frozen source/payload and
native tests. The [donor validation](provenance/donor_validation.json) independently
checks all 90 camera frames against pinned training parquets. The unused first
deployment was canceled before uploading a payload or making model calls.
