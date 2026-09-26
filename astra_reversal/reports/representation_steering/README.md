# Representation steering: evidence, next experiment and rollout videos

[Open the portable HTML report and video gallery](index.html). The 11 clips are original recorded MP4s, total 642,434 bytes. No simulator replay, model call or transcoding was used. [Video provenance](gallery.json).

The new seed-47 representation screen is an **assisted-correction experiment** with no policy updates. It has no audited results in this report. Its [plan](../../REPRESENTATION_PLAN.md) and [protocol](../../configs/representation_steering_v1.json) specify 20 known OOD tasks, one native baseline and at most two revisions per arm; maximum 420 rollouts. The three-case seed-19 pilot has maximum 63. These are planned budgets.

| Completed study | Separate baseline | Measured outcome | Budget / scope |
|---|---:|---|---|
| Phase interpolation, seed 29 | 7/20 | Astra TEI 11/20; TLI 13/20; TLI + annotations 13/20; random noise 9/20 | Shared baseline + up to 2 revisions; 20 known tasks, reset 0 |
| Real pixel perturbations, seed 37 | 8/20 | Astra occlusion 12/20; donor blend 12/20; random occlusion 10/20, random blend 11/20 | Shared baseline + up to 2 revisions; separate 20-reset cohort |
| FRS development, seed 19 | 0/3 native and repeated-noise controls | Direct 2/3; FRS 2/3; critique/no-learning 0/3; final learned noise 0/3 | Three selected known failure tasks, evaluation reset 1 |

All 18 FRS development judgments were ties; no candidate was promoted and no auxiliary update ran. Its physical cost remains 45 rollouts, 775 provider calls and at least 3,229,704 tokens (3 unknown-usage calls). This does not establish that a trained actor failed to learn. The [full frozen FRS run](https://us-west-2-aws.osmo.nvidia.com/workflows/astra-pi05-frs-evaluation-20260925-1) is **RUNNING** in the saved snapshot; its 1,740 rollouts are planned coverage, and no partial efficacy is published here.

The older seed-7 native baseline is 86/200, with TF32 enabled, fresh noise and ten trials per task. It remains [historical context](../ood_baseline.json), not a comparator pooled with the three cohorts above. All measured numbers are copied from complete audited reports and bound in [study_summary.json](study_summary.json).

- [Research evidence and concrete hypotheses](RESEARCH.md)
- [Exact method, observation access, effort and feedback boundaries](METHODS.md)
- [New exact system prompt and settings](prompts/representation.json)
- [Earlier phase prompt](prompts/phase_interpolation.json), [pixel prompt](prompts/image_perturbations.json), [four FRS prompts](prompts/frs.json)
- [Completed phase results](../phase_interpolation/evaluation/results/report.md), [pixel results](../image_perturbations/evaluation/index.html), [FRS development](../frs_policy_improvement/development/html/index.html)
- [Gallery sources and timing](gallery.json), [all published file hashes](manifest.json)

Videos are outcome-selected examples, including failures and matched baselines; they do not estimate rates. They show the original raw external camera **before each action**, at 20 fps, omitting stabilization, inference pauses and the final post-action image. Standard LIBERO is an ID task example with unknown exact checkpoint training overlap. Simulator success need not imply stable final placement. Historical outcomes, setup costs and ongoing studies are never pooled.

Snapshot: 2026-09-26T18:51:13.482683+00:00. No dollars, 100% reliability, novel-task generalization or RL-efficiency advantage is claimed.
