# Completed three-task FRS development study

The direction methods each passed **2/3** separate-reset development episodes.
Both native baselines, the critique-and-FRS method, and the auxiliary-policy
checkpoints passed **0/3**. All three task recordings passed complete audits.
These are three selected, known failure cases at seed 19, with adaptation on
reset 0 and evaluation on reset 1. They cannot estimate performance across all
20 OOD tasks. The full 20-task evaluation uses seed 43 and ten evaluation resets
per task; its results remain pending.

Open the [HTML report with diagrams and plots](html/index.html), the
[detailed Markdown report](report.md), [exact result JSON](report.json), or
[episode table](episodes.csv). The [original bundle manifest](manifest.json)
binds the generated report and evidence. This README, behavioral review and
CUDA setup evidence are supplementary; they do not alter the compiled results.

## Outcomes and online cost

| Task | Native full / repeated noise | Direct Astra direction | Astra flow reversal | Critique and FRS | Learned noise |
|---|---|---|---|---|---|
| Wine bottle in bowl | Fail / fail | Pass | Pass | Fail | Fail |
| Milk on plate | Fail / fail | Fail | Pass | Fail | Fail |
| Center bowl on cabinet | Fail / fail | Pass | Fail | Fail | Fail |

The successful cases differ between direct steering and flow reversal. Equal
2/3 totals do not establish equivalence or superiority. No policy that selects
the successful method retrospectively was evaluated.

| Task | Method | Outcome | Executed actions | Astra calls | Reported tokens |
|---|---|---|---:|---:|---:|
| Wine bottle in bowl | Direct direction | Pass | 103 | 11 | 15,633 |
| Wine bottle in bowl | Flow reversal | Pass | 78 | 8 | 11,385 |
| Milk on plate | Direct direction | Fail at cap | 300 | 30 | ≥42,296 |
| Milk on plate | Flow reversal | Pass | 83 | 9 | 12,900 |
| Center bowl on cabinet | Direct direction | Pass | 202 | 21 | 29,751 |
| Center bowl on cabinet | Flow reversal | Fail at cap | 300 | 30 | 42,422 |

These costs cover each single online evaluation rollout. Every ten actions,
Astra either selects a direction or defers to the fresh native prediction.
The failed milk/direct rollout includes an HTTP 503 with unknown token usage.
The full physical study totals are **45 rollouts, 12,766 actions, 14,720 native
velocity evaluations, 775 provider calls, and at least 3,229,704 tokens**.
There were 771 accepted responses and four failed/rejected calls; two timeouts
and the HTTP 503 lack usage. The wrong-request-ID rejection's known usage is
included. No dollar price is inferred.

## Why the learning loop did not improve here

Both critique arms ran all three prescribed revisions on each task. None
rescued its adaptation baseline, and all 18 pairwise judgments were `same`.
Consequently there were **zero promotions and zero experimental optimizer
updates**. The learned actor retained its exact native-noise fallback through
all three checkpoints. There is no observed learning gain in this cohort.

The action editor proposed 77 edits, 550 native deferrals and three rejected
calls across 630 requests. All 77 edits occurred during adaptation. The final
no-learning evaluations had no promoted rules and therefore deferred on all
90 requests. Those evaluations measure the complete rule-promotion procedure,
including its empty-rule fallback. The paper-direction role made 53 steering
decisions, 55 deferrals and one rejected call across 109 requests.

The [independent behavioral review](behavior/README.md) checks provider/event
bindings, costs, the calibrated direction transform and selected camera views.
One recorded action edit asks for ten closed-gripper actions; inverse/forward
flow produces six closing rows followed by four reopening rows. Action-reference
editing is approximate after flow reversal and noise transformation. Its
execution fidelity and the conservative rule gate are concrete limitations to
investigate in a separately specified follow-up.

## What the success flag establishes

The released wine-bottle task uses an `On` predicate, despite the displayed
instruction saying “in the bowl.” Its contact/relative-position checks do not
require gripper release or sustained placement. The two successful terminal
camera pairs show the bottle at the bowl with the gripper still close around
it. They establish recorded benchmark success; stable released placement was
not separately tested. The behavioral review preserves the exact task/source
hashes and raw image examples.

## Independent CUDA setup validation

Because no development rollout was admitted for learning, a separate OSMO L40S
job exercised the production auxiliary learner. Two synthetic 1,000-step fits
produced identical parameter, optimizer and replay hashes, and checkpoint
reloads reproduced predictions. The [raw setup result](cuda_setup/probe.json)
and [validation summary](cuda_setup/validation.json) preserve the frozen source
and runtime bindings. This validates execution of the CUDA training path;
it supplies no task-success result or Astra promotion.

Its separate overhead was 10.52 seconds of fitting, 15.58 seconds for the probe,
and 76.76 seconds from OSMO initialization through completion (~0.02132 GPU-hours).
It made zero provider calls or native-model velocity evaluations and is excluded
from the 45-rollout study totals above.
