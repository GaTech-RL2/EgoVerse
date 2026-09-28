# Rollout and intervention speed

These are recorded wall-clock measurements on the OSMO L40S workers. They describe this synchronous research harness, not an optimized real-robot controller. No new rollouts or inference calls were made for this summary.

## Experiment A: online Astra

Astra is queried at actions 0, 25, …, 275 in the language and image studies: at most twelve calls per 300-action attempt. The simulator waits for each response. The selected edit is used between queries; the policy replans every five executed actions. A response can select an ineffective edit or be rejected, so call frequency is not the frequency of effective edits.

**Astra client latency averages 9–11 seconds per call.** This timer includes payload construction, network/provider wait and response validation. It is not isolated provider inference time. Means include physical calls whose proposals were rejected or whose requests failed.

| Cohort | Method | Mean client seconds/call (calls) | Median seconds for 300 actions (rollouts) | Mean seconds/physical attempt |
|---|---|---:|---:|---:|
| language | Native baseline (recovered noise) | — (0) | 42.8 (13) | 40.1 |
| language | Astra TEI | 9.41 (270) | 166.3 (21) | 165.8 |
| language | Astra TLI | 11.08 (209) | 196.8 (15) | 174.1 |
| language | Astra TLI + annotations | 10.79 (200) | 198.1 (14) | 169.9 |
| pixels | Native baseline (recovered noise) | — (0) | 42.4 (12) | 38.6 |
| pixels | Astra occlusion | 10.35 (218) | 190.0 (16) | 210.4 |
| pixels | Astra demonstration-image blend | 11.28 (222) | 187.5 (17) | 172.2 |

The 300-action medians condition on attempts that reached the cap; earlier successes are excluded from that column. The last column includes all actual attempts of that method, including early successes. Online rows contain rescue attempts only, with the shared native baseline counted once in its own row. These are descriptive timings, not paired estimates of the cost of an edit. Language/image native timings include one-time noise initialization and numerical checks; they should not be treated as a steady-state policy benchmark.

Including each arm's native baseline and all retries through success or the cap, mean summed rollout time per evaluated case was:

| Cohort | Method | Mean seconds/case | Matched baseline seconds/case |
|---|---|---:|---:|
| language | Astra TEI | 247.4 | 40.1 |
| language | Astra TLI | 222.9 | 40.1 |
| language | Astra TLI + annotations | 210.0 | 40.1 |
| pixels | Astra occlusion | 249.0 | 38.6 |
| pixels | Astra demonstration-image blend | 219.4 | 38.6 |

These case averages include baseline successes requiring no Astra calls and failed retries. They exclude work outside the rollout timer, worker setup and queue time. Summing parallel workers gives measured work, not experiment elapsed time.

## Experiment B: deployment without online Astra

All methods have forty OOD task/reset cases and zero online Astra calls. The policy replans every five actions. The learned selector refreshes every twenty-five actions; the selected operator is applied on intervening replans.

| Method | Mean policy callback ms/replan | Median seconds for 300 actions (rollouts) | Mean seconds/rollout, all 40 cases | Executed actions/wall second |
|---|---:|---:|---:|---:|
| Native baseline | 370 | 28.9 (25) | 23.3 | 9.87 |
| Recorded teacher schedule | 383 | 28.9 (13) | 17.9 | 9.48 |
| Learned TEI/TLI selector | 395 | 30.3 (21) | 21.7 | 9.45 |
| Learned flow head | 373 | 29.4 (26) | 23.4 | 10.02 |
| Selector-gated flow head | 383 | 30.1 (24) | 22.7 | 9.75 |

The policy callback includes preparation, editing, sampling, decoding and callback logging. Its weighted mean is total callback seconds divided by all replans, not a GPU kernel microbenchmark. Different trajectories and intervention activity prevent attributing the whole difference to selector/head overhead. The recorded schedule's shorter mean episode mainly reflects fewer executed actions after success; it does not show a faster policy. Teacher acquisition and training costs are separate.

## FRS: three-case development evaluation

This pilot queried Astra every ten actions. A non-deferred FRS edit adds ten inverse and ten forward vector-field evaluations, after the native ten-step prediction: thirty evaluations on an edited replan versus ten on native/deferred replans. The published pilot summaries do not isolate per-call client latency, so no FRS inference-only timing is inferred from whole-rollout time.

| Case | Native seconds / actions | FRS seconds / actions | FRS calls / edits | FRS simulator success |
|---|---:|---:|---:|---|
| libero_goal_ood:seed19:task6:state1 | 17.9 / 300 | 63.6 / 78 | 8 / 2 | Yes |
| libero_spatial_ood:seed19:task2:state1 | 19.1 / 300 | 80.4 / 83 | 9 / 5 | Yes |
| libero_spatial_ood:seed19:task8:state1 | 18.5 / 300 | 225.4 / 300 | 30 / 11 | No |

All three native attempts failed. FRS succeeded earlier on two cases, so their durations do not compare equal amounts of executed motion. These three cases do not establish timing or efficacy across all twenty tasks; the larger FRS run was interrupted.

## What the videos and timers mean

The simulator's configured control frequency is 20 Hz. Three hundred actions correspond to fifteen seconds of simulated control; every twenty-five actions corresponds to 1.25 simulated seconds. A 9–11-second synchronous Astra call is much longer than that interval. The recorded 20-fps videos omit inference pauses and cannot demonstrate wall-clock execution speed. Even the native recipe harness averaged roughly ten executed actions per wall second including its reset, recording and simulation work; these measurements do not establish real-time 20-Hz operation.

The completed speed result is that offline schedule reuse and the learned selector retain near-native rollout timing while avoiding online Astra waits. Sparse event-triggered or asynchronous Astra intervention could reduce waiting, but that scheduling change has not been evaluated here.

## Sources and regeneration

- [language source report](../phase_interpolation/evaluation/results/report.json)
- [pixels source report](../image_perturbations/evaluation/results/report.json)
- [frs source report](../frs_policy_improvement/development/report.json)
- [recipe source report](../learned_correction_recipe/results/report.json)
- [Timer boundaries in the shared runner](../../intervention_rollout.py)
- [Exact aggregates and source hashes](runtime.json)

With the repository environment activated, run `python astra_reversal/reports/smart_system2_results/build_runtime.py`, then `python astra_reversal/reports/smart_system2_results/build_results.py` to refresh the publication manifest.
