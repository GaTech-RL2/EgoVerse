# Coverage and prior measurements

This snapshot verifies all **20 released LIBERO-OOD compositions**. The new FRS study has no measured results yet. The separate seed-37 image evaluation is pending complete recording and audit; its final success rate is not asserted here.

The [OOD paper, §4](https://arxiv.org/html/2505.03500v5#S4) reports ten runs per task: 100 per suite, 200 total. The pinned release uses environment seed 7 once per task followed by ten resets. Its prescribed-state loading is commented out, and no OOD `.pruned_init` assets occur in the pinned tree. Consequently, state indexes denote captured reset-stream positions.

Release: [`587a6cbf64f16c7b87fa5805dc0ed934192239a4`](https://github.com/QuanyiLi/pi0-text-latent/tree/587a6cbf64f16c7b87fa5805dc0ed934192239a4). The task order below is release order 0; spelling, including “bbq source”, is preserved. File hashes and BDDL goals are in [coverage.json](coverage.json).

| Suite | ID | Exact instruction | Native seed-7 successes /10 |
|---|---:|---|---:|
| Goal | 0 | put the cream cheese in the basket | 4 |
| Goal | 1 | put the orange juice on the stove | 7 |
| Goal | 2 | put the bbq source on the plate | 5 |
| Goal | 3 | put the tomato sauce on top of the cabinet | 0 |
| Goal | 4 | put the wine bottle on the stove | 10 |
| Goal | 5 | put the wine bottle on the plate | 8 |
| Goal | 6 | put the wine bottle in the bowl | 3 |
| Goal | 7 | put the cream cheese on the plate | 2 |
| Goal | 8 | put the cream cheese on the stove | 0 |
| Goal | 9 | put the cream cheese on top of the cabinet | 8 |
| Spatial | 0 | put the butter on the plate | 10 |
| Spatial | 1 | put the chocolate pudding on the plate | 10 |
| Spatial | 2 | put the milk on the plate | 4 |
| Spatial | 3 | put the orange juice on the plate | 10 |
| Spatial | 4 | put the bowl on cookie box on the stove | 0 |
| Spatial | 5 | put the bowl on cookie box on the cabinet | 0 |
| Spatial | 6 | put the bowl next to the plate on the cabinet | 0 |
| Spatial | 7 | put the bowl next to the plate on the stove | 0 |
| Spatial | 8 | put the bowl at table center on the cabinet | 4 |
| Spatial | 9 | put the bowl at table center on the stove | 1 |

The repaired native Euler-10/TF32-on baseline is **86/200**: Goal 47/100 and Spatial 39/100, with zero canonical execution errors. The original recording had 82 successes and 8 hardware execution errors. All 25 episodes in the affected shard were replaced, including previously valid outcomes; it was not a failure-only retry. Its repair wall latency is not comparable because CUDA_LAUNCH_BLOCKING=1 was enabled. [Source](../ood_baseline.json).

| Separate measured study | Reset coverage | Outcomes |
|---|---|---|
| Matched prior RK4/TF32-off study | Seed 7, ten resets/task, 200 per method | Fresh 91/200; reused 90/200; Astra reversal 5/200 |
| Static intervention search | Seed 19, reset 0, 20 cases; up to 4 revisions | Common baseline 8/20; random 10/20; Astra noise 9, language 9, vision 9, noise+language 9, noise+vision 10, language+vision 9, joint 8 |
| Phase interpolation | Seed 29, reset 0, 20 cases | Common baseline 7/20; random 9; oracle TEI 10, TLI 12, combined 13; Astra TEI 11, TLI 13, TLI+vision 13 |
| Pixel intervention development | Seed 19, three known cases; up to 2 revisions | Common baseline 0/3; random noise 0/3; random occlusion 1/3; random blend 1/3; Astra occlusion 1/3; Astra blend 1/3 |
| Pixel intervention evaluation | Seed 37, reset 0, 20 expected cases | Pending complete audit; no final result |

The static study used visual annotations and a pooled input-text residual. Phase interpolation used TEI and layerwise TLI with optional temporary visual marks. The newer pixel study uses real donor-image blends or neutral occlusion. These operators and denominators are not pooled. Oracle phase arms used one revision; Astra/random phase arms used up to two. All retry results include a shared baseline and simulator reset access, with failures retained. [Static report](../iterative_interventions/evaluation/report.json); [phase report](../phase_interpolation/evaluation/results/report.json); [pixel development](../image_perturbations/development/results/report.json).

Success follows the released predicates, not a separate visual judgment. In particular, the wine-in-bowl BDDL uses `On`; object `check_ontop` combines height, contact and a 0.1 m horizontal-distance threshold, while site geometry uses a 0.10 m vertical band. The exact source links and each task’s BDDL digest are preserved in coverage.json. A binary terminal flag does not establish stable release or agreement with a VLM’s assessment.

The [FRS paper, §4.1 and Appendix B/D/E](https://arxiv.org/html/2606.13675v2) uses Euler 10 reverse/forward, executes 10 actions per query, and evaluates 50 trials/task on different standard LIBERO task sets. Finite-step displacement is intentional. Its calibrated plumb line belongs to the VLM input; policy cameras remain raw. Its noise-policy learning and zero-shot steering are distinct from the earlier high-accuracy RK4 inversion experiments.

The proposed [frozen FRS protocol](../../configs/frs_policy_improvement_v1.json) uses seed 43, adaptation reset 0 and evaluation resets 1–10 for each of the 20 known compositions. It separates native full/repeated-noise controls, direct Astra directions, Astra FRS, critique without learning and learned noise. Three fixed adaptation revisions are separate from learned-checkpoint evaluation after each round. Development remains the three known seed 19 cases. This is new-reset evaluation on known tasks, not novel-task generalization; checkpoint training overlap is unknown. No superiority over RL efficiency is tested.

All numeric entries above come from completed source reports whose exact bytes are SHA-bound in coverage.json. Current FRS/pixel evaluation placeholders are not measurements.
