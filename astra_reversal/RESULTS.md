# Measured LIBERO and flow-reversal results

Measured on 2026-09-24 using OSMO L40S GPUs. The selected checkpoint is frozen; no training or weight conversion was performed. This snapshot contains completed policy baselines, initial numerical gates, and a genuine Astra development rollout that failed numerical validation and task completion. Paired RK4 OOD controls and Stage 1 OOD evaluation remain in progress; no Astra success-rate improvement or Stage 2 result is reported. Compact evidence is tracked in [reports/](reports/README.md).

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

There is no known generating policy noise for Astra's actions. The preflight's known-noise result belongs to a separate policy-generated sample. See the [numerical audit](NUMERICAL_RESULTS.md). Paired OOD fresh-noise, reused-noise, and Astra-reversal work is in progress, with final results pending. The current experiment concerns Stage 1; Stage 2 augmentation has no measured result here.

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

See the exact [worker summary](reports/astra_development_smoke.json), [review and provenance checks](reports/astra_development_review.json), and unchanged [rollout video](reports/astra_development_smoke.mp4). The full 102,180,346-byte archive is preserved under the recorded workflow prefix. OOD runs retain their frozen settings, and their task scores will be reported together with runtime reconstruction failures. No tolerances or prompts were changed to remove this negative result.

## Validation

Fresh-main validation passed 468 repository unit tests, with five existing optional OpenPI tests skipped. All seven native LeRobot integration tests passed separately in the pinned runtime. Astra contributes 161 unit tests plus those seven integration tests; the extracted standalone payload also passed both suites. Each GPU run verifies its hardware family and exact checkpoint hashes. Numerical gates also match the controller specification, checkpoint/input provenance, solver, resolution, and time grid. The [read-only runtime inversion audit](audit_astra_inversions.py) checks recorded Stage 1 array hashes, reconstruction under the same condition and latent, and subsequent latent reuse with advancing observations; it performs no additional GPU solves. Snapshot hashes and exact copy/compaction rules are recorded in [snapshot_sources.json](reports/snapshot_sources.json).
