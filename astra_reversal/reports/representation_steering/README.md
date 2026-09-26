# Representation steering: evidence, next experiment and rollout videos

[Open the portable HTML report and video gallery](index.html). The 11 clips are original recorded MP4s, total 642,434 bytes. No simulator replay, model call or transcoding was used. [Video provenance](gallery.json).

The new seed-47 representation screen is an **assisted-correction experiment** with no policy updates. It has not launched. Its [plan](../../REPRESENTATION_PLAN.md) and [protocol](../../configs/representation_steering_v1.json) specify 20 known OOD tasks, one native baseline and at most two revisions per arm; maximum 420 rollouts. The three-case seed-19 pilot had maximum 63. These are protocol budgets, not completed coverage.

## Interrupted pilot: verified observations, incomplete cohort

The [seed-19 initial results and seven original videos](development/results/index.html) are now archive/provider bound. A spending-cap failure stopped the pilot; the full seed-47 evaluation **has not launched**. Two cases have verified partial archives and one has a complete case seal. A seal does not make unexercised Astra modes valid integration evidence. No three-case or twenty-task efficacy rate is released.

There are 42 completed physical rollouts and 11,849 completed actions. Two unfinished rollouts add at least 300 actions; their outcomes remain unknown. Recorded compute is 24,700 velocity evaluations, including 390 setup/probe evaluations once per case. This includes 600 unfinished evaluations beyond the 24,100 completed-rollout total.

The complete recorded ledger contains 220 calls / 98 accepted, at least 1,129,046 reported tokens and 122 unknown-usage calls. All 122 failed calls returned HTTP 429; 48 explicitly carried the safe type `budget_exceeded`. Missing usage is not zero. Account-wide spending is not attributed entirely to this experiment.

Two descriptive completed observations are wine TEI revision 2 (98 actions) and milk TLI revision 1 (120 actions), after their native baselines failed. They are not cohort rates or causal estimates. The Spatial-8 baseline succeeded in 106 actions; forced native retry, random TLI/VEI/VLI and Astra TEI/TLI/VEI then failed, while forced VLI succeeded in 125 actions with 50 recorded representation-effect actions. These extras test stability and are not rescues.

The combined TLI+VLI and pixel-blend modes had no accepted decisions. They were never started on wine/milk; their Spatial-8 rollouts were native fallback only. Wine/milk VEI and VLI completed prefixes likewise have no accepted Astra edits. [Exact completed and unfinished accounting](development/results/report.md) and [raw recording provenance](development/results/video_gallery.json) keep those limitations visible.



| Completed study | Separate baseline | Measured outcome | Budget / scope |
|---|---:|---|---|
| Phase interpolation, seed 29 | 7/20 | Astra TEI 11/20; TLI 13/20; TLI + annotations 13/20; random noise 9/20 | Shared baseline + up to 2 revisions; 20 known tasks, reset 0 |
| Real pixel perturbations, seed 37 | 8/20 | Astra occlusion 12/20; donor blend 12/20; random occlusion 10/20, random blend 11/20 | Shared baseline + up to 2 revisions; separate 20-reset cohort |
| FRS development, seed 19 | 0/3 native and repeated-noise controls | Direct 2/3; FRS 2/3; critique/no-learning 0/3; final learned noise 0/3 | Three selected known failure tasks, evaluation reset 1 |

All 18 FRS development judgments were ties; no candidate was promoted and no auxiliary update ran. Its physical cost remains 45 rollouts, 775 provider calls and at least 3,229,704 tokens (3 unknown-usage calls). This does not establish that a trained actor failed to learn. The [full frozen FRS run](../frs_policy_improvement/EVALUATION_INTERRUPTION.md) was also **interrupted by the spending cap**: 952 completed recorded rollouts, 6,587 calls and at least 15,562,019 known tokens, with 1,166 unknown-usage calls. Its 1,740 rollouts were planned coverage. Only five of twenty tasks completed and were fully audited; no complete-cohort efficacy is published. In that execution-selected [five-task subset](../frs_policy_improvement/evaluation_gate_observations.json), 30 judgments were 29 same / 1 uncertain, with zero promotions or auxiliary updates. Those gate observations remain separate from the three-case development cohort.

The older seed-7 native baseline is 86/200, with TF32 enabled, fresh noise and ten trials per task. It remains [historical context](../ood_baseline.json), not a comparator pooled with the three completed cohorts above. Complete-cohort numbers and incomplete archive-bound observations have separate sources and scope in [study_summary.json](study_summary.json).

- [Research evidence and concrete hypotheses](RESEARCH.md)
- [Exact method, observation access, effort and feedback boundaries](METHODS.md)
- [New exact system prompt and settings](prompts/representation.json)
- [Earlier phase prompt](prompts/phase_interpolation.json), [pixel prompt](prompts/image_perturbations.json), [four FRS prompts](prompts/frs.json)
- [Completed phase results](../phase_interpolation/evaluation/results/report.md), [pixel results](../image_perturbations/evaluation/index.html), [FRS development](../frs_policy_improvement/development/html/index.html)
- [Gallery sources and timing](gallery.json), [all published file hashes](manifest.json)

Videos are outcome-selected examples, including failures and matched baselines; they do not estimate rates. They show the original raw external camera **before each action**, at 20 fps, omitting stabilization, inference pauses and the final post-action image. Standard LIBERO is an ID task example with unknown exact checkpoint training overlap. Simulator success need not imply stable final placement. Historical outcomes, setup costs and ongoing studies are never pooled.

Snapshot: 2026-09-26T19:50:05.107887+00:00. No dollars, 100% reliability, novel-task generalization or RL-efficiency advantage is claimed.
