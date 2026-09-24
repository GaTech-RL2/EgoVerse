# Recorded-condition numerical validation

The subsequent genuine closed-loop Astra development rollout **failed** the
unchanged roundtrip limit on 2 of 28 proposals: maximum full internal errors
were 0.459375 and 0.118617 at steps 330 and 485. Their conditions and recovered
latents match exactly. The task also failed at 520 actions, independently of
this numerical result. See the [development audit](reports/astra_development_review.json).
The initial passing checks below therefore do not establish reconstruction
accuracy for arbitrary Astra endpoints.

A separate [recorded-proposal replay](osmo/astra_proposal_replay.py) fixes one
passing development control at step 310 and both failures at steps 330 and 485,
then measures all three at RK4 resolutions 100, 200, and 500. The
[predeclared input plan](reports/astra_development_replay_plan.json) binds all 21
input arrays to their original requests, provider responses, and conditions.
CPU replay of the stored endpoints reproduced all three original metrics exactly.
The GPU diagnostic is pending; it makes no new agent calls, selects no solver,
and leaves the frozen OOD configuration unchanged.

Cubic RK4 with **100 steps** passed all 14 recorded LIBERO-10 development conditions and was selected on 2026-09-24. The tracked [L40S numerical summary](reports/runtime_numerics.json) covers one task-0/state-0 trajectory at observation steps 0, 20, …, 260, rather than 14 independent episodes. The checkpoint is frozen and uses the OpenPI LIBERO input profile. The summary preserves all 42 candidate/condition metric rows, native parity, source/checkpoint identities, costs, and endpoint hashes; it omits repeated grids and is not a replacement for the complete worker gate. [Snapshot provenance](reports/snapshot_sources.json) identifies the original report by SHA-256.

Each candidate regenerated an endpoint from the recorded full `[1, 10, 32]` noise tensor, inverted that endpoint, and regenerated actions under the same condition. All camera/state/prompt identities and known-noise hashes were checked. The grid is `t_j = (j/N)^3`, reversed for generation. The unchanged gates require maximum absolute known-noise error ≤ **0.1**, full internal and decoded controller reconstruction errors ≤ **0.02**, and native-sampler parity error ≤ **1e-5**. Decoded reconstruction is measured before clipping.

| RK4 steps | Velocity evaluations per solve | Maximum noise error | Maximum internal action error | Maximum decoded action error | Passing conditions |
|---:|---:|---:|---:|---:|---:|
| **100** | **400** | **0.000335217** | 1.86265e-7 | 2.38371e-7 | **14/14** |
| 200 | 800 | 0.000432014 | 1.19209e-7 | 2.38371e-7 | 14/14 |
| 500 | 2,000 | 0.000980854 | 1.19209e-7 | 2.38371e-7 | 14/14 |

These are maxima over all conditions and tensor elements. The three worst prior RK4/50 conditions were screened first; every candidate passed screening and completed all 14. Selection used the fewest velocity evaluations per solve among candidates passing every condition. No OOD outcomes or control success entered selection. An independent review recomputed the JSON aggregates and acceptance decisions and matched the known-noise hashes against the recorded inputs.

Native parity was checked separately on every condition: **10-step Euler adapter versus the actual LeRobot native Euler sampler**, with maximum decoded difference **0** in all 14 cases. This establishes sampler-adapter parity for those checks; it does not assert that RK4 and Euler produce identical actions. Checkpoint, tokenizer, normalization, and model/input source identities matched the recorded source. The run explicitly disabled TF32 and used float32 flow tensors.

Increasing resolution did not consistently reduce the float32 residuals. At observation step 100, maximum noise errors were 0.000157237, 0.000428081, and 0.000309348 for N=100/200/500. The overall maximum also increased across these resolutions. These measurements establish the stated tolerance gate, not a convergence order or an isolated explanation for the residuals; no precision ablation identifies their cause.

The [original RK4/50 rollout audit](checkpoints/libero_l40s_runtime_recovery.json) remains **3/14 passing**, with maximum noise error **0.630013645**. The [earlier initial-observation diagnostic](checkpoints/libero_l40s_fixed_numerics.json) reported 0.0514342 at RK4/100, whereas the new step-0/RK4/100 result is 0.000193059. Thus the difference between runs cannot be attributed solely to increased resolution. The old fixed worker did not record the TF32 setting; the exact source of the cross-run difference has not been isolated.

The tracked [genuine Astra GPU preflight](reports/astra_proposal_preflight.json) **passed**. Maximum direct controller reconstruction error was **1.32135e-7**, measured without clipping; maximum full internal reconstruction error was **1.41561e-7**. Its [implementation](osmo/proposal_probe.py) verifies the pinned genuine request, accepted numeric response, and provider record, including reconstructed RGB/state fingerprints, schema, raw response identity, and actual model `azure/openai/gpt-6-astra`. Their original file hashes are retained in the tracked report; raw files are excluded from Git. It requires the selected 14-condition gate and additionally checks direct decoded replay against 0.02. The separate policy-generated known-noise check had maximum error **0.000193059**: Astra's proposed actions have no known generating policy noise. The actual proposal's inverse and forward solves used 800 velocity evaluations and took 8.76 seconds together. The probe replayed a saved genuine response and made no new Astra call.

The OOD native Euler-10 baseline uses TF32 on, while the three matched RK4/100 conditions use TF32 off. Comparisons against the native baseline therefore include a solver and runtime change. The matched fresh-noise, reused-noise, and genuine Astra-reversal conditions share the numerical settings and frozen scenes; final results remain pending in this snapshot.

Passing the initial numerical checks establishes reconstruction only for their recorded inputs. It does not establish useful proposals, successful closed-loop control, performance after conditioning changes, or OOD improvement. The live development smoke recorded 26/28 passing same-condition roundtrips and two failures; all 76 later generations reused the intended latent on fresh observations, with zero fallbacks. The first seven normalized channels contain the worst errors, while padding maxima are below 0.000877. No cause or solver correction has yet been isolated, and no Stage 2 measurement is reported. The [runtime inversion audit](audit_astra_inversions.py) separately checks reconstruction for each recorded Astra proposal under identical conditioning and latent, then verifies latent reuse on later observations without treating changed-condition output as reconstruction error.

The [report index](reports/README.md) explains the compact snapshots and links optional local raw artifacts, including the complete numerical gate and original vision-smoke records.
