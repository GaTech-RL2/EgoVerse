# Observed phase interpolation: experiment record

The frozen π0.5 controller has completed all **three audited development cases**
with real online Astra decisions. Astra TLI and TLI plus vision rescued all three
within two revisions; all three native controls failed on these cases. The
**20-task, seed-29 evaluation is running** and has no complete result yet.
Development uses previously inspected compositions and is not a generalization
success-rate estimate.

The implementation is described in [PHASE_INTERPOLATION.md](../../PHASE_INTERPOLATION.md).
The [bank record](banks/README.md) verifies nine standard training donors,
180 demonstrations and 18,947 frames. The rollout payload is source commit
`4e54e9ae8f55cc4e663e599e4bd19bb94add2baa`, SHA256
`5b51adf8a604ec68461a8b8933d11f9b9f2b7cd9295eb463b715272f3cc7c842`.

## Complete development comparison

Every arm starts from the same failed baseline on each case. A revision is a
full rollout from that same reset; decisions within a rollout are separate.
Oracle arms receive one revision. Random noise and Astra receive up to two and
stop at first success. All outcomes below passed full-array and feedback audits.

| Intervention | Success within 1 revision | Success within 2 revisions | Provider tokens across all attempts |
| --- | ---: | ---: | ---: |
| Random noise | 0/3 | 0/3 | 0 |
| Oracle TEI | 1/3 | Capped at 1 | 0 |
| Oracle TLI | 2/3 | Capped at 1 | 0 |
| Oracle TEI+TLI | 2/3 | Capped at 1 | 0 |
| Astra TEI | 1/3 | 1/3 | 423,684 |
| Astra TLI | 2/3 | 3/3 | 182,412 |
| Astra TLI+vision | 2/3 | 3/3 | 192,997 |

The complete development study used **37 physical rollouts, 8,702 actions,
21,290 velocity evaluations, 105 provider calls and 799,093 provider tokens**.
Three proposals were rejected and their costs are included. The shared baseline
is counted once in physical totals. All 10,986 stored arrays were audited.

TLI rescued milk in one revision with four decisions and **21,929 tokens**;
it rescued wine in two revisions with 17 decisions and **132,062 tokens**.
The latter includes the failed 12-decision revision. The cabinet example below
used one revision, five TLI decisions and **28,421 tokens**. TEI's unsuccessful
wine and milk arms each exhausted two revisions and 24 decisions; their cost
remains in the table.

The [complete tables, curves and provenance](development/results/report.md)
retain case-level outcomes, controls, budgets and cost. The
[milk failure/rescue example](development/behavior/README.md) compares actual
TEI and TLI feedback. Each arm sees only its own earlier attempts and the common
baseline; TLI did not receive the TEI failures.

## Cabinet development example

On **center bowl → cabinet**, seed 19, the recovered-noise, known-noise and
native fresh-noise controls all failed within 300 actions. Results below are
one reset of one previously inspected composition, not an aggregate success rate.

| Intervention | Outcome within cap | Revisions to success | Astra decisions | Provider tokens |
| --- | --- | ---: | ---: | ---: |
| Random noise | Failed after two revisions | — | 0 | 0 |
| Oracle TEI | Failed after one revision | — | 0 | 0 |
| Oracle TLI | Succeeded | 1 | 0 | 0 |
| Oracle TEI+TLI | Succeeded | 1 | 0 | 0 |
| Astra TEI | Succeeded | 1 | 4 | 21,668 |
| Astra TLI | Succeeded | 1 | 5 | 28,421 |
| Astra TLI+vision | Succeeded | 1 | 5 | 29,078 |

Astra TEI selected the center-bowl and cabinet donors and changed α from
0.3 → 0.3 → 0.85 → 1.0 at actions 0, 25, 50 and 75. It observed the failed
baseline's plate placement, waited through approach, then increased the cabinet
weight around grasp and lift. The task succeeded after 100 actions in the
intervention rollout. These are recorded choices and outcomes, not an isolated
causal test of its verbal explanation.

[Camera and decision examples](development/examples/README.md) retain the actual
feedback images and vision edits. The [array audit](development/audits/worker_2_arrays.json)
and [feedback receipt](development/audits/worker_2_receipt.json) passed, including
all 2,976 arrays, 470 policy generations, 14 provider decisions and 206 decoded
feedback-camera bindings. The complete case used 11 physical rollouts, 2,341
actions and 79,167 provider tokens across its three Astra arms. Language choices
differ across TLI and TLI+vision; their difference does not isolate vision's effect.

## Infrastructure interruption

The original development workflow was
`astra-pi05-interpolation-development-20260925-1`. Its cabinet worker completed.
The wine and milk workers subsequently terminated with OSMO control failures
while assigned to the same host. Scheduler events record repipelining before
termination; the underlying infrastructure cause is not established.

Only the interrupted workers were rerun to completion in
`astra-pi05-interpolation-development-recovery-20260925-2`, using the identical
payload, bank inventory and reset protocol. The completed cabinet result was
retained. The rerun used normal scheduling priority and excluded the interrupted
host. The first recovery deployment was canceled after a missing storage driver
prevented one container from mounting its filesystem; neither worker ran the
study. The second recovery's real worker metadata matches the original inputs.

The [interruption record](development/interruption/README.md) preserves **63
provider usage records and 519,361 tokens** of additional overhead. Published
events also prove one further accepted decision whose usage record was not
preserved. Its tokens, other unuploaded work and in-flight calls remain unknown.
These costs are separate from completed-case comparisons and are not silently
discarded. The retained cabinet worker's cost is counted once.

No monetary provider rate is available. Reported reasoning tokens are a subset
of output tokens. The separate transport-only smoke used 2,550 tokens and is
excluded from the rollout numbers above.

## Evaluation in progress

`astra-pi05-interpolation-evaluation-20260925-1` was submitted only after all
development integration, numerical, array and feedback audits passed. It uses
eight OSMO L40S workers, the identical frozen payload and donor banks, and all
20 prescribed goal/spatial OOD compositions with seed 29. Each composition has
one reset. Successful baselines require no intervention; failed baselines enter
the same capped comparisons above. No intervention setting was changed after
development outcomes were observed.
