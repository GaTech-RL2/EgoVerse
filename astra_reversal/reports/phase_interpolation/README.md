# Observed phase interpolation: experiment record

The frozen π0.5 controller has completed one audited development case with real
online Astra decisions. The three-case development study and the 20-task,
seed-29 evaluation are **not yet complete**. The numerical protocol and donor
library are frozen; this page will be updated with the complete results.

The implementation is described in [PHASE_INTERPOLATION.md](../../PHASE_INTERPOLATION.md).
The [bank record](banks/README.md) verifies nine standard training donors,
180 demonstrations and 18,947 frames. The rollout payload is source commit
`4e54e9ae8f55cc4e663e599e4bd19bb94add2baa`, SHA256
`5b51adf8a604ec68461a8b8933d11f9b9f2b7cd9295eb463b715272f3cc7c842`.

## Completed development example

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

Only the interrupted workers are being restarted in
`astra-pi05-interpolation-development-recovery-20260925-1`, using the identical
payload, bank inventory and reset protocol. The completed cabinet result is
retained. The rerun uses normal scheduling priority and excludes the interrupted
host. Recorded additional overhead from the interrupted workers is **at least
63 provider calls and 519,361 tokens**. Unuploaded work and in-flight calls remain
unknown. These costs are separate from completed-case comparisons and are not
silently discarded.

No monetary provider rate is available. Reported reasoning tokens are a subset
of output tokens. The separate transport-only smoke used 2,550 tokens and is
excluded from the rollout numbers above.
