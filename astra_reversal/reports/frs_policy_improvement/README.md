# Astra flow reversal and policy improvement

Corrected **development is complete and independently audited on three known
failure cases at seed 19**. On each task's separate evaluation reset 1, direct
Astra directions succeeded on 2/3 cases and Astra flow reversal steering on
2/3. Both native controls, critique without learning, and the final learned-noise
checkpoint scored 0/3. These selected development cases are not the full
20-task evaluation.

The [development HTML report](development/html/index.html) and
[detailed Markdown report](development/report.md) retain every prescribed
outcome, task audit, checkpoint, failed call and cost. Both adaptation arms had
0/3 cumulative successes after three revisions; Astra promoted no candidates,
so there were **zero auxiliary updates or optimizer steps**. The completed
development recordings contain 45 unique rollouts, 12,766 actions, 14,720
velocity evaluations and 775 provider calls. Known usage is **at least
3,229,704 tokens**; three calls have unknown usage. This includes the completed
cohort's failed calls and counts the shared baseline once. Setup and interrupted
attempts remain separate.

The full evaluation was **interrupted when the inference key reached its spending cap**. All eight L40S workers stopped and retained their final archives. 5 of 20 planned task records had complete independent audits; no all 20-task FRS success rate is available. The [interruption and cost report](EVALUATION_INTERRUPTION.md) records 6,587 calls and at least 15,562,019 reported tokens, including failed calls. Its planned 1,740 rollouts were not all completed. A funded inference credential is needed to continue.

The [HTML method report](index.html) brings together the completed development
summary, current evaluation snapshot, all 20 task names, two flow diagrams,
the detailed approach and all four exact Astra prompts. The
[publication manifest](method_report_manifest.json) binds its source narratives,
development evidence and diagrams to exact hashes. The
[infrastructure record](infrastructure.json) preserves the earlier startup
interruptions and queue timeout, marks development completed, and records the
full evaluation launch separately. Those infrastructure interruptions are not
measured task failures.

All **20 released OOD tasks** have already been tested in earlier studies. The
[coverage inventory](COVERAGE.md) gives every task and the historical native
baseline: 86/200 successes across ten resets per task. The separate completed
[image report](../image_perturbations/evaluation/README.md) and
[standalone HTML](../image_perturbations/evaluation/index.html) cover all 20 tasks
at seed 37/reset 0: common baseline 8/20, Astra occlusion 12/20 and Astra demo
blending 12/20 within two revisions. Different seeds, execution settings and
retry budgets keep those results separate from this FRS experiment.

The [detailed approach](APPROACH.md) explains what Astra sees, when it intervenes,
action normalization, inverse/forward flow, judgment-gated replay and the
auxiliary policy update. The [prompt appendix](PROMPTS.md) contains all four
exact system-message strings and their hashes, with the original
[worker manifest](prompts.json). The
[frozen protocol](../../configs/frs_policy_improvement_v1.json) compares native
full/repeated noise, direct Astra directions, paper-like Astra FRS, critique
without learning, and learned noise. Each learned checkpoint is evaluated after
its corresponding adaptation round; the three-round cap is fixed. The runtime
remains commit `a50f92dfd18a7fdfe1c6198d34ddb43386d28396`, payload SHA256
`0828e19a85a7474d2e6ef2bbe705406da51690cdb4ed6891687407a007fc5bae`.

The initial prompt-v1 development was deliberately interrupted after identifying
an editor-capability mismatch. Its [retained evidence and costs](development_v1_interruption.json)
include 151 recorded provider calls and at least 694,705 tokens, with three
interrupted calls of unknown usage. These are additional costs, excluded from
the completed corrected-development cohort and from efficacy. No full FRS
evaluation estimate or learning-improvement claim is made from partial runs.
