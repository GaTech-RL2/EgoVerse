# Measured LIBERO and flow-reversal results

Measured on 2026-09-24 using OSMO L40S GPUs. The completed paired OOD evaluation scored **5/200 for genuine Stage 1 Astra reversal**, versus **91/200 for matched fresh noise** and **90/200 for matched reused noise**. Its runtime audit found **39/3,108** same-condition roundtrips above the unchanged numerical limit, despite verified provider bindings and latent reuse. The selected checkpoint and experiment settings stayed frozen; no training or weight conversion was performed. A separate development replay reproduces the earlier numerical failures. This is a negative result for the tested configuration, with no Stage 2 or paper-reproduction claim. Compact evidence is tracked in [reports/](reports/README.md).

## Weight loading

- Hugging Face revision: `a217bfd3b14673cf2ce597e69997ab21866438dd` (`lerobot/pi05_libero_base`).
- All checkpoint files match the pinned SHA-256 inventory. Strict parameter loading succeeds, including the declared shared embedding alias.
- All parameters are frozen. Adapter actions and LeRobot native-sampler actions match exactly on the recorded development observation (`max_abs = 0`).
- The full internal CPU/CUDA endpoint difference is at most `7.152557e-7`.

See the [weight verification](checkpoints/libero_l40s_weight_verification.json).

## Full baseline: 455/500 = 91.0%

All ten LIBERO-10 tasks completed 50 prescribed-state episodes, with the standard 520-action budget and ten stabilization steps. There were zero execution errors, fallbacks, or Astra calls. Every episode ID was verified against one frozen manifest.

| Task | Instruction | Successes |
|---|---|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 49/50 |
| 1 | put both the cream cheese box and the butter in the basket | 50/50 |
| 2 | turn on the stove and put the moka pot on it | 47/50 |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 46/50 |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 48/50 |
| 5 | pick up the book and place it in the back compartment of the caddy | 49/50 |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 47/50 |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 50/50 |
| 8 | put both moka pots on the stove | 22/50 |
| 9 | put the yellow and white mug in the microwave and close it | 47/50 |

This run uses the selected LeRobot PyTorch/float32 export, OpenPI input settings, environment seed 0, and explicit NumPy noise keyed by seed/step/draw, with a common schedule across prescribed states. Full published OpenPI model and random-stream parity has not been established. Native sampler parity here means this adapter versus LeRobot's actual sampler with identical explicit noise.

The first reset of tasks 0–2 was used to debug the input pipeline and is included in the standard 500-state manifest. With identical weights and those three resets, the exported settings scored 0/3 and the OpenPI settings scored 3/3. The changed inputs are horizon 10, plain task tokenization, the official quantile statistics, and the example image resize. These settings were restored as a group; their individual effects were not ablated.

See the [full report](checkpoints/libero_l40s_full_baseline.json), [paired input pilot](checkpoints/libero_l40s_development_result.json), and [OSMO run](https://us-west-2-aws.osmo.nvidia.com/workflows/astra-pi05-libero-baseline-20260924-1). Complete lossless artifacts are archived under the report’s `artifact_prefix`.

## LIBERO-OOD baseline: 86/200 = 43.0%

The frozen pi05 checkpoint completed all 20 tasks from the [paper's released LIBERO-OOD environment](https://github.com/QuanyiLi/pi0-text-latent/tree/587a6cbf64f16c7b87fa5805dc0ed934192239a4): ten trials per task, 300 actions, ten stabilization steps, and seed 7. The native Euler-10 fresh-noise baseline enables TF32. Goal-OOD scored 47/100 and Spatial-OOD scored 39/100. There were zero execution errors, zero-action successes, fallbacks, or Astra calls in the canonical result. Its 43,052 executed actions used 86,430 velocity evaluations.

The original run recorded 82/200 successes and eight execution errors following a CUDA launch failure. The entire affected 25-episode Goal shard was rerun on a fresh L40S, including its previously successful episodes. That shard changed from 10 successes/eight errors to 14 successes/zero errors. Both raw runs are retained, and replacement did not depend on success. The replacement uses the same frozen dynamic states and randomized fixture transforms; their restoration was checked on the GPU. Its `CUDA_LAUNCH_BLOCKING=1` debugging setting makes its wall time unsuitable for a controlled timing comparison.

This is a new pi05 baseline on the released tasks. The paper evaluates a different pi0 policy, and checkpoint training overlap with these tasks is unknown. It is not a reproduction of the paper's pi0 or TLI results. See the [validated baseline report](reports/ood_baseline.json), [per-task results](reports/ood_baseline_tasks.csv), and [frozen protocol](OOD_PLAN.md). The report preserves original/repair counts, manifest hashes, workflow identities, and the timing caveat.

## Initial flow diagnostics: historical solver selection

The unchanged acceptance limits are maximum internal/decoded action error 0.02, maximum known-noise error 0.1, and native Euler parity error 1e-5. These initial measurements used one real development observation and full `[1,10,32]` tensors. Their solver selection is superseded by the 14-condition gate below.

| Solver / grid | Resolution | Maximum noise error | Maximum decoded action error | Passed |
|---|---:|---:|---:|---|
| Uniform Euler | 10 steps | 6.44486 | 0.0142508 | False |
| Uniform Heun | 200 steps | 1.84012 | 0.00190888 | False |
| Cubic Heun | 200 steps | 0.0763079 | 1.51366e-05 | True |
| Cubic RK4 | 50 steps | 0.0584911 | 5.68515e-05 | True |
| Adaptive RK45 | rtol 1e-6, atol 1e-8 | 0.0180501 | 1.50128e-05 | True |

The cubic grid is `t_j=(j/N)^3`, reversed for forward generation. RK4/50 was initially selected by the fewest velocity evaluations among passing development candidates: 200 evaluations per solve. Its measured forward latency in the first control pilot is 2.31 seconds, versus 27.3 seconds for the passing adaptive reference solve. DOP853 exceeded its 20,000-evaluation cap; that failed diagnostic is retained. The broader validation below supersedes that initial solver choice.

See [fixed-grid diagnostics](checkpoints/libero_l40s_fixed_numerics.json) and [adaptive diagnostics](checkpoints/libero_l40s_adaptive_numerics.json). An [optional local convergence figure](artifacts/osmo_dev_l40s/reversal_convergence_all.png) is available with the downloaded raw artifacts.

## Policy-reference control pilot

All four methods succeeded on the same prescribed task-0 reset, with zero fallbacks or execution errors. These are single-episode development results.

| Method | Actions | Episode seconds | Velocity evaluations |
|---|---:|---:|---:|
| Fresh noise | 247 | 126.3 | 10,000 |
| Reused noise | 269 | 135.0 | 10,800 |
| Reference inversion + reuse | 264 | 198.6 | 16,200 |
| Matched computation | 232 | 173.2 | 14,200 |

The inversion ablation regenerated actions from fresh observations every five environment steps and refreshed its policy reference/inversion every twenty steps. It used no Astra proposals or observation augmentation.

**Historical runtime limitation:** only 3/14 known-noise inversions met the 0.1 maximum-error tolerance. Maximum noise error was 0.6300 (median per-plan maximum 0.2734), while same-condition action reconstruction remained within 0.0001001. Thus the initial-observation gate did not establish accurate noise recovery across the rollout. RK4/50 is no longer the selected runtime solver; the expanded development validation below addresses those same 14 conditions.

See the [paired control report](checkpoints/libero_l40s_control_pilot.json) and [runtime array audit](checkpoints/libero_l40s_runtime_recovery.json). An [optional local inversion rollout video](artifacts/osmo_fixed_l40s/extracted/results/inversion_only/episode_00000.mp4) is available with the raw artifacts. All referenced arrays were hash-verified; local post-processing reproduced the worker’s audit. No success-rate improvement is established by one episode.

## Expanded numerical validation and actual Astra

All 14 saved standard-LIBERO development conditions passed cubic-grid RK4 at 100, 200, and 500 steps with the original tolerances. These are observations from one task-0/state-0 trajectory at steps 0, 20, …, 260. The frozen selection rule chooses 100 steps, or 400 velocity evaluations per solve. Its maximum known-noise error was 0.00033522; native Euler parity error was zero on every condition. Both TF32 flags are disabled for this gate and the matched steering conditions. All 42 condition/candidate metric rows and provenance are in the [compact numerical report](reports/runtime_numerics.json); workers continue to require the complete original gate. The change from the earlier run cannot be attributed solely to step count: TF32 was explicitly disabled in the new run, and the cause of the cross-run difference has not been isolated.

The authenticated NVIDIA endpoint returned a valid Stage 1 proposal from `azure/openai/gpt-6-astra` for a real two-camera development observation, and authenticated vision inference also succeeded from the OSMO worker. The [genuine-proposal GPU preflight](reports/astra_proposal_preflight.json) passed with maximum direct controller replay error **1.32135e-7**, before clipping, and maximum full internal reconstruction error **1.41561e-7**. The tracked report preserves model identity, request/response/provider hashes, and the measured errors. The original images, numeric response, and provider usage remain in the [optional local vision-smoke artifacts](artifacts/astra_vision_smoke).

There is no known generating policy noise for Astra's actions. The preflight's known-noise result belongs to a separate policy-generated sample. See the [numerical audit](NUMERICAL_RESULTS.md). The completed OOD results below concern Stage 1; Stage 2 augmentation has no measured result here.

The Euler baseline enables TF32 while the matched RK4 conditions disable it. Comparisons against Euler therefore include solver and runtime changes. The three matched conditions share those numerical settings and frozen reset states, enabling paired assessment of noise reuse and Astra steering without that solver/runtime difference. Numerical reconstruction alone does not establish control quality or OOD improvement.

## Genuine development rollout: numerical and task failures

The full standard LIBERO-10 task-0/state-0 smoke used seed 7 and the frozen RK4/100 configuration. It completed 520 actions without an execution exception or fallback, but the task was unsuccessful and no subgoal was completed. Its 32 actual Astra calls yielded 28 accepted proposals; four exhausted-subgoal responses were rejected and regenerated. Episode wall time was 1,169.21 seconds, with 52,800 velocity evaluations. This single development episode is not a benchmark success-rate estimate.

All 28 inverse/first-forward pairs used identical recorded conditions and exact recovered latents. The worker verified 856 array files. All 104 forward generations retained the intended latent, including 76 later generations with advancing steps and changed observation IDs. Independent local checks reproduced the binding of every accepted proposal to the original provider response and configured model.

The unchanged maximum full-internal-error limit of 0.02 passed on **26/28** proposals and failed on two:

| Observation step | Maximum full internal error | Per-plan RMSE |
|---:|---:|---:|
| 330 | 0.459374666 | 0.028102330 |
| 485 | 0.118616998 | 0.006661244 |

These errors are in normalized model coordinates, including all 32 channels; they are not decoded controller errors or known-noise recovery measurements. The same failures appear in the first seven channels, while padding errors remain below 0.000877. The workflow returned `complete_with_issues` / exit 2 for this numerical negative. The cause has not been isolated. The initial policy-generated gate and one-proposal preflight were insufficient to establish coverage for this live rollout.

See the exact [worker summary](reports/astra_development_smoke.json), [review and provenance checks](reports/astra_development_review.json), and unchanged [rollout video](reports/astra_development_smoke.mp4). The full 102,180,346-byte archive is preserved under the recorded workflow prefix. OOD runs retained their frozen settings, and their task scores are reported below together with runtime reconstruction failures. No tolerances or prompts were changed to remove this negative result.

## Recorded development proposals: fixed-resolution replay

The predeclared replay used the passing proposal at step 310 and the two failures at steps 330 and 485 from that same standard development episode. All three N100 recovered latents and first-forward endpoints reproduced bit-for-bit. The L40S worker completed all nine pairs at fixed N100/N200/N500, using 18 solves and 19,200 velocity evaluations; the probe took 285.56 seconds including model loading and report synchronization.

| Development step | N100 maximum full internal error | N200 | N500 |
|---:|---:|---:|---:|
| 310 | 1.10269e-6 | 3.57628e-7 | 4.47035e-8 |
| 330 | 0.459375 | 0.387576 | 0.313804 |
| 485 | 0.118617 | 0.0183960 | 0.00192770 |

Step 330 fails the unchanged 0.02 threshold at every tested resolution; its N500 decoded-controller maximum is also 0.313741. Step 485 passes at N200 and N500. The worker's `complete_reproduced` / exit 0 status confirms completed reproduction, not a generally valid numerical method. The [independent archive review](reports/astra_development_replay.json) checked all 21 source arrays, 36 output arrays, nine error/cost rows, grids, original provider bindings, and runtime identities. Its full report and 1,226,183-byte archive are identified by SHA-256.

This is a targeted development characterization of one passing control and two known failures, not an unbiased endpoint sample or a new control evaluation. It made no new Astra calls or environment rollouts, used no OOD data, and selected no replacement solver. Residual reductions do not establish an asymptotic convergence order or the exact cause of failure. The original OOD settings remain frozen.

## Completed paired OOD evaluation

All four methods cover the same 200 frozen OOD episode IDs, with ten trials for each of 20 tasks, a 300-action budget, and seed 7. The three matched methods use cubic RK4/100 with TF32 disabled. All methods have zero execution errors and zero-action successes in the canonical result.

| Method | Goal-OOD | Spatial-OOD | Total | Executed actions | Velocity evaluations | Controller calls |
|---|---:|---:|---:|---:|---:|---:|
| Native Euler-10 fresh noise, TF32 on | 47/100 | 39/100 | 86/200 (43.0%) | 43,052 | 86,430 | 0 |
| Matched fresh noise | 51/100 | 40/100 | 91/200 (45.5%) | 42,368 | 3,402,800 | 0 |
| Matched reused noise | 51/100 | 39/100 | 90/200 (45.0%) | 43,451 | 3,491,600 | 0 |
| Genuine Astra reversal | 5/100 | 0/100 | 5/200 (2.5%) | 59,308 | 5,988,400 | 3,565 |

The reversal score includes its configured policy fallback. One of its five successful episodes, Goal task 6/state 5, executed 20 fallback actions at steps 120–139 of 169 total actions; the other four successes had none. At that refresh, an invalid schema/episode response was followed by a schema-valid response rejected because its subgoal was already completed or had exhausted its recovery budget. This episode remains in the method's full denominator and is not described as wholly Astra/latent-controlled. See the [success-context audit](reports/ood_reversal_success_context.json).

Reversal minus matched fresh noise is **−43.0 percentage points**, with a 95% paired bootstrap interval of **[−48.0, −38.0]** points. Reversal minus reused noise is −42.5 points [−47.5, −37.0]. Reused minus fresh noise is −0.5 points [−5.5, 4.5], which does not establish a benefit from reuse. Resampling uses initial states within each task, 2,000 draws, and seed 7. These are results for the frozen checkpoint, proposal configuration, reset states, and single environment seed; the experiment does not identify a general failure of every reversal method.

The [complete report](reports/ood_paired.json), [method table](reports/ood_paired_methods.csv), [80 method/task rows](reports/ood_paired_tasks.csv), and [800 paired episode rows](reports/ood_paired_episodes.csv) preserve all outcomes. The original baseline and entire-shard repair remain separate and unchanged. The earlier [completed-control snapshot](reports/ood_matched_controls.json) is retained. The report's `valid_complete_evaluation` field verifies execution checks; it does not assert passing numerical validation. [Fixed episode videos](reports/README.md#fixed-episode-videos) show the same episode ID for every method, without model/API waiting time.

## Complete Stage 1 runtime audit

The read-only [final audit](reports/ood_runtime_audit.json) verifies all **25 archives, 32 reversal shards, 200 episodes, and 96,066 array files**. Every one of the 3,108 accepted Astra proposals is bound to its original provider response and paired with the first forward generation under the identical condition and exact recovered latent. **3,069 pairs pass and 39 fail** the unchanged maximum full-internal-error limit of 0.02. The largest error is **1.8645510077**. These failures occur in 30 episodes; all 39 also exceed 0.02 in the first seven normalized action channels, while padding remains within tolerance.

All 11,754 recorded proposal-based generations use the intended recovered latent. The 8,646 later generations have advancing steps and changed observation IDs, with no latent mismatch. Those changed-condition outputs test reuse and are not reconstruction errors. Astra actions have no known generating policy noise. Full/action/padding maxima and RMSE for every failing plan remain in the audit. Correct binding and reuse do not remove numerical failures or establish control quality, and these measurements do not isolate the cause of the low task score.

The [provider review](reports/ood_provider_review.json) distinguishes **3,565 recorded calls**, **3,433 schema-accepted responses**, **3,108 controller-accepted proposals**, **457 controller rejections**, and **429 configured retries**. The rejections comprise 325 completed/exhausted-subgoal responses and 132 backend failures: 112 wrong action-specification IDs, 11 schema/episode errors, five HTTP 503s, two HTTP 500s, and two transport timeouts. Actual execution used **58,763 latent actions and 545 fallback actions** (0.919% fallback), across 109 fallback refreshes. Both matched controls and the Euler baseline have zero fallbacks or Astra calls. All accepted provider bindings are verified, with zero missing provider records or recorded model/settings mismatches. These checks are independent of numerical accuracy and task success. Retry/fallback consequences remain in the review; no manual retries or protocol changes were made.

The [archive provenance](reports/ood_archive_provenance.json) retains whole-object SHA-256/ETag/length receipts, per-unit audit hashes, frozen reset/config/runtime identities, and offline validator results. One local postprocessor dependency omission was corrected by copying the unchanged `ActionSpec` dependency and rerunning the read-only checks; it was not an experiment, integrity, or numerical error. Per-execution API usage and latency summaries are retained without inferring a monetary invoice or pooled latency p95. The native baseline's solver/TF32 difference and repair timing caveat still apply.

## Iterative noise, language-embedding, and vision interventions

The later [iterative-intervention study](reports/iterative_interventions/README.md)
completed all 20 seed-19 OOD cases on eight OSMO L40S workers. The shared
recovered-noise baseline succeeded on 8/20; random-noise search and Astra's
noise + vision arm each reached 10/20 within four revisions, on different rescued
cases. Other single/pair Astra arms reached 9/20, and the all-three arm remained
8/20. No Astra arm exceeded random search at the cap. Some semantic edits rescued
individual cases in one revision; the report retains exact iterations, failed
searches, and all token costs.

Evaluation used 323 calls and 1,406,726 provider-reported tokens, including seven
rejected proposals, plus 424 physical rollouts and 267,580 velocity evaluations.
All eight raw archives and 151,076 saved arrays passed read-only checks. This
follow-up uses a policy-generated reference and iterative edits with simulator
reset access. It is separate from Stage 1's numeric-action inversion results
above and does not establish unseen training compositions or a recipe for 100%
success.

## Validation

Fresh-main validation passed 468 repository unit tests, with five existing optional OpenPI tests skipped. All seven native LeRobot integration tests passed separately in the pinned runtime. Astra contributes 161 unit tests plus those seven integration tests; the extracted standalone payload also passed both suites. Each GPU run verifies its hardware family and exact checkpoint hashes. Numerical gates also match the controller specification, checkpoint/input provenance, solver, resolution, and time grid. The [read-only runtime inversion audit](audit_astra_inversions.py) checks recorded Stage 1 array hashes, reconstruction under the same condition and latent, and subsequent latent reuse with advancing observations; it performs no additional GPU solves. Snapshot hashes and exact copy/compaction rules are recorded in [snapshot_sources.json](reports/snapshot_sources.json).
