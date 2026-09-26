# Astra flow reversal and policy improvement

The corrected FRS study has **no completed development or evaluation result
yet**. Its unchanged development run is queued on GROOT L40S-01 as
[`astra-pi05-frs-development-20260925-5`](https://us-west-2-aws.osmo.nvidia.com/workflows/astra-pi05-frs-development-20260925-5).
Two recent L40S-03 starts were reclaimed by higher-priority shared GPU quota
enforcement during startup. These are infrastructure interruptions, not measured
task failures. The [infrastructure record](infrastructure.json) preserves their
scope and completed preflight checks.

All **20 released OOD tasks** have already been tested in earlier studies. The
[coverage inventory](COVERAGE.md) gives every task and the historical native
baseline: 86/200 successes across ten resets per task. The separate completed
[image report](../image_perturbations/evaluation/README.md) and
[standalone HTML](../image_perturbations/evaluation/index.html) cover all 20 tasks
at seed 37/reset 0: common baseline 8/20, Astra occlusion 12/20 and Astra demo
blending 12/20 within two revisions. Different seeds, execution settings and
retry budgets keep those results separate from this new experiment.

The [detailed approach](APPROACH.md) explains exactly what Astra sees, when it
intervenes, all four prompt contracts, action normalization, inverse/forward
flow, judgment-gated replay and the auxiliary policy update. The
[frozen protocol](../../configs/frs_policy_improvement_v1.json) compares native
full/repeated noise, direct Astra directions, paper-like Astra FRS, critique
without learning, and learned noise. It uses one adaptation reset and ten
separate evaluation resets per task. Each learned checkpoint is evaluated after
its corresponding adaptation round; the three-round cap is fixed.

The complete evaluation plan contains 200 evaluation episodes per method at the
final round, 400 additional evaluations of the first two learned checkpoints,
and 140 physical adaptation rollouts: **1,740 planned rollouts**. These counts
are planned coverage, not completed measurements. Full evaluation follows
complete development recordings and their independent audits. The experiment
runtime remains commit `a50f92dfd18a7fdfe1c6198d34ddb43386d28396`, payload SHA256
`0828e19a85a7474d2e6ef2bbe705406da51690cdb4ed6891687407a007fc5bae`.

The initial prompt-v1 development was deliberately interrupted after identifying
an editor-capability mismatch. Its [retained evidence and costs](development_v1_interruption.json)
include 151 recorded provider calls and at least 694,705 tokens, with three
interrupted calls of unknown usage. It is excluded from efficacy. Exact worker
prompts, failed-call usage, capped failures, iteration counts and training costs
will accompany the complete audited FRS report; partial recordings are not
released as a success-rate estimate.
