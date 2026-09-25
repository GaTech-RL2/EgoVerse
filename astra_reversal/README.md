# Astra reversal

Implementation of the supplied [research brief](SPEC.md), using a frozen pi0.5
policy and genuine Astra proposals. Experiment code, isolated dependencies, and
records live in this directory; existing training code is unchanged.

The separate [iterative intervention protocol](INTERVENTIONS.md) adds explicit
rollout feedback and revision of noise, actual language embeddings, and visual
annotations. It records attempts to first success and provider token usage for
each intervention and combination. The completed Stage 1 results below did not
use that iterative search and must not be presented as its evaluation.

**Status on 2026-09-24:** the paired OOD evaluation is complete. Genuine Stage 1
Astra reversal scored **5/200**, compared with **91/200** for matched fresh noise
and **90/200** for matched reused noise. The complete runtime audit found
**39/3,108** same-condition roundtrips above the unchanged 0.02 limit, despite
verified provider bindings and recovered-latent reuse. The initial development
gates passed but did not establish runtime coverage or useful control. This is
a negative result for the tested configuration; no Stage 2 result is reported.
One of the five successful reversal episodes included 20 policy-fallback
actions; the other four had none. The score measures the configured controller
with its logged fallback behavior.

| Completed measurement | Result | Tracked evidence |
|---|---|---|
| Standard LIBERO-10, Euler-10 fresh noise | 455/500 successes (91.0%); zero execution errors | [Baseline report](checkpoints/libero_l40s_full_baseline.json) |
| Released LIBERO-OOD, Euler-10 fresh noise, TF32 on | 86/200 (43.0%): Goal 47/100, Spatial 39/100; zero canonical execution errors | [Repaired baseline report](reports/ood_baseline.json), [20-task table](reports/ood_baseline_tasks.csv) |
| Matched OOD controls, cubic RK4/100, TF32 off | Fresh noise 91/200; reused noise 90/200; zero execution errors or fallbacks | [Completed controls](reports/ood_matched_controls.json) |
| Genuine Astra OOD reversal, cubic RK4/100, TF32 off | 5/200 (2.5%): Goal 5/100, Spatial 0/100; 545/59,308 actions were fallback | [Paired outcomes](reports/ood_paired.json), [task table](reports/ood_paired_tasks.csv) |
| Complete OOD runtime audit | 3,069/3,108 roundtrips within 0.02; maximum error 1.864551; all accepted proposal bindings and latent reuses verified | [Numerical audit](reports/ood_runtime_audit.json), [provider review](reports/ood_provider_review.json) |
| Recorded-condition flow gate, TF32 off | Cubic RK4/100 passed all 14 development conditions; maximum known-noise error 0.000335217 | [Numerical summary](reports/runtime_numerics.json) |
| Genuine Astra proposal, cubic RK4/100 | Direct controller replay maximum error 1.32135e-7 before clipping | [Proposal preflight](reports/astra_proposal_preflight.json) |
| Genuine closed-loop development rollout | 26/28 roundtrips within 0.02; worst internal error 0.459375. Task failed at 520 actions, with zero execution errors/fallbacks | [Development audit](reports/astra_development_review.json), [video](reports/astra_development_smoke.mp4) |
| Fixed replay of three genuine development proposals | All N100 endpoints/latents reproduced bit-for-bit; step 330 exceeds 0.02 at N100, N200, and N500 | [Replay review](reports/astra_development_replay.json) |

The OOD baseline replaces the entire 25-episode shard affected by a CUDA failure,
including its previously successful episodes. Its original and replacement
counts are retained. Standard and OOD scores use different protocols and success
predicates and must remain separate. These are new pi0.5 measurements on released
tasks; checkpoint training overlap is unknown, and they do not reproduce the
paper's pi0 or TLI results. See [RESULTS.md](RESULTS.md),
[NUMERICAL_RESULTS.md](NUMERICAL_RESULTS.md), and the
[tracked report index](reports/README.md).

## Policy and input provenance

The selected weights are
[`lerobot/pi05_libero_base`](https://huggingface.co/lerobot/pi05_libero_base/tree/a217bfd3b14673cf2ce597e69997ab21866438dd),
pinned to revision `a217bfd3b14673cf2ce597e69997ab21866438dd`.
The [checkpoint inventory](checkpoints/lerobot_pi05_libero_base.json) records all
six downloaded files. Strict loading and frozen parameters are checked; no
training or weight conversion is performed.

The saved LeRobot export has a 50-step model horizon, 32 internal channels, state
tokens, and empty normalization statistics. The measured LIBERO runs explicitly
use `policy.input_profile: openpi_libero`: horizon 10, seven controller channels,
plain task tokens, 224px image resizing, and the official quantile statistics.
This preserves the selected weights and native velocity while changing the
input convention. With identical weights and three development resets, the
saved export settings scored 0/3 and the OpenPI input profile scored 3/3. Those
three resets are also included in the standard 500-episode manifest. The input
changes were applied together and have not been individually ablated.

The [input asset inventory](checkpoints/openpi_libero_input_assets.json),
[tokenizer inventory](checkpoints/paligemma_tokenizer.json), and
[weight verification](checkpoints/libero_l40s_weight_verification.json) preserve
the relevant identities. Model states are full `[1,10,32]` tensors. The policy
executes five environment actions per generated ten-action chunk. Native parity
compares this adapter against the actual LeRobot sampler with identical noise;
it does not establish full published OpenPI model or random-stream parity.

Local model, tokenizer, and reference assets live under `.deps/` and stay outside
Git. Install the pinned [LeRobot requirements](requirements-lerobot.txt) and
[LIBERO requirements](requirements-libero.txt) in an isolated Python 3.11+
environment. The live controller check supports the pinned robosuite 1.4.1.

## Genuine Astra Stage 1

[astra_client.py](astra_client.py) calls the authenticated endpoint
`https://inference-api.nvidia.com/v1/chat/completions` with the configured model
`azure/openai/gpt-6-astra`. Authentication comes from
`NVIDIA_INFERENCE_API_KEY`; credentials are injected into workers and are excluded
from payloads and reports. Accepted provider responses must identify that exact
model and complete normally.

The [OOD driver](osmo/ood_steering.py) freezes low reasoning effort,
`max_completion_tokens=8192`, JSON output, cache bypass, and no specified
temperature. It uses a 170-second HTTP timeout and a 180-second controller
request timeout, one regeneration after invalid output, and an Astra refresh
period of 20 environment steps. The same settings apply to the
[development integration smoke](osmo/astra_dev_smoke.py).

Astra receives both current RGB views, eight-channel robot proprioception, the
full instruction, controller specification, and bounded observed history. It
supplies every value in a `[10,7]` continuous action chunk plus subgoal metadata.
Simulator object poses, success predicates, demonstrations, and policy-generated
action proposals are absent from its request. Responses are schema-checked and
bound to the exact observation and request fingerprint. Raw accepted and
rejected provider responses, model identity, token usage, and latency are
preserved in the run archive.

Stage 1 encodes and pads the genuine proposal to `[1,10,32]`, inverts it under the
current observation and full instruction, and reuses the recovered latent for
forward generation on fresh observations. Proposals are never interpolated or
filled in by the policy. Invalid or stale proposals receive one regeneration
attempt, then an explicitly logged full-instruction policy fallback with fresh
noise. Decoded policy actions use the shared logged bounds rule.

The [genuine proposal preflight](reports/astra_proposal_preflight.json) validates
one saved real two-camera development request, its original numeric response,
provider identity, and file hashes before running the GPU round trip. Astra
actions have **no known generating policy noise**. The report's known-noise metric
comes from a separate policy-generated sample on the same observation. A passing
preflight establishes numerical reconstruction under fixed conditioning; it
does not establish useful proposals or successful closed-loop control.

The subsequent [development episode](reports/astra_development_smoke.json) made
32 real Astra calls and accepted 28 plans. Four exhausted-subgoal responses were
rejected and regenerated. All 104 generations used the recorded recovered
latent, including 76 fresh-observation reuses. Nevertheless, proposals at steps
330 and 485 failed the unchanged full internal reconstruction limit, with
maximum errors 0.459375 and 0.118617. The audit verified their identical
conditioning and exact latent hashes. These are normalized model-space errors;
they are separate from the preflight's decoded controller metric. The earlier
14 policy-generated conditions and one genuine proposal did not establish
coverage for arbitrary Astra endpoints. The OOD evaluation retained its frozen
configuration and reports its numerical failures alongside all task outcomes.

## Numerical gate and paired OOD controls

The selected solver is RK4 with 100 steps on `t_j=(j/N)^3`, reversed for forward
generation, with TF32 disabled and float32 flow tensors. Each solve uses 400
velocity evaluations. RK4/100, /200, and /500 all passed the 14 recorded
development conditions; selection used the fewest evaluations among candidates
passing every condition. The limits remain maximum internal/decoded action
error 0.02, maximum known-noise error 0.1, and native Euler parity error 1e-5.
Solver selection used no OOD outcomes.

These 14 conditions are observations from one standard-LIBERO task-0/state-0
development trajectory. They are not 14 independent episodes. The earlier
RK4/50 gate passed on its initial observation but only 3/14 runtime inversions;
that selection is superseded. The new gate also explicitly disables TF32, so
the improvement cannot be attributed solely to increased step count. Larger
step counts did not monotonically reduce the observed float32 residuals.

The frozen [OOD plan](OOD_PLAN.md) compares:

| Condition | Solver and runtime | Proposal/noise source | Status in this snapshot |
|---|---|---|---|
| Native Euler baseline | Euler 10, TF32 on | Fresh Gaussian noise, full task | Complete: 86/200 |
| Matched fresh-noise control | Cubic RK4 100, TF32 off | Fresh Gaussian noise, full task | Complete: 91/200 |
| Matched reused-noise control | Cubic RK4 100, TF32 off | Reused Gaussian noise, full task | Complete: 90/200 |
| Genuine Astra reversal | Cubic RK4 100, TF32 off | Inverted Astra proposal, then latent reuse | Complete: 5/200 |

All methods replay the same frozen OOD scenes, including dynamic state and
randomized fixture transforms. Comparing the native baseline with RK4 conditions
includes both solver and TF32 changes. The three matched conditions share these
settings and permit paired assessment of noise reuse and Astra steering.
Additional SPEC controls and Stage 2 augmentation remain outside the completed
measurements reported here.

Reversal minus matched fresh noise is −43.0 percentage points, with a 95%
paired bootstrap interval of [−48.0, −38.0] points. All four methods completed
their 200 assigned episodes with zero execution errors or zero-action successes.
The report's `valid_complete_evaluation` field concerns those execution checks;
it does not mean numerical validation passed. The [archive audit](reports/ood_archive_provenance.json)
verifies all 25 archives and 32 reversal shards. Numerical failures occurred in
30 episodes; these measurements do not isolate the cause of the poor task score.

The tracked [compact numerical report](reports/runtime_numerics.json) is for
review. Workers require the complete original `runtime_diagnostics.json`, whose
SHA-256 is recorded in the summary; the compact file cannot replace that gate.
Reversal workers also require the genuine-proposal preflight and exact checkpoint,
controller, solver, grid, and resolution identities.

## Benchmark protocol and execution

Use separate isolated source checkouts for standard LIBERO and the OOD fork:

| Suite | Pinned environment | Actions + stabilization | Trials/task | Reset source |
|---|---|---:|---:|---|
| `libero_10` | `Lifelong-Robot-Learning/LIBERO@f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` | 520 + 10 | 50 | Prescribed initial states |
| `libero_goal_ood` | `QuanyiLi/pi0-text-latent@587a6cbf64f16c7b87fa5805dc0ed934192239a4` | 300 + 10 | 10 | Frozen released seeded reset stream |
| `libero_spatial_ood` | Same OOD revision | 300 + 10 | 10 | Frozen released seeded reset stream |

The standard baseline used environment seed 0; OOD uses the released default
seed 7. Explicit NumPy noise is keyed by episode seed, environment step, and draw
index, repeating its schedule across initial states. This differs from an
uninterrupted policy RNG stream. OOD retains the released modified success
predicates, including its relaxed contact and stove rules.

The runner verifies environment revisions and imported paths, checks the live
controller contract, and writes `LIBERO_CONFIG_PATH` inside the run directory.
It does not modify `~/.libero/config.yaml`. Standard `--libero-root` is
`astra_reversal/.deps/libero`; OOD uses
`astra_reversal/.deps/libero-ood/third_party/modified_libero`.

OSMO entrypoints and resource specifications:

| Work | Entrypoint | Workflow |
|---|---|---|
| Standard 500-episode baseline | [benchmark.py](osmo/benchmark.py) | [Eight L40S GPUs](osmo/benchmark_l40s.yaml) |
| OOD baseline | [ood_benchmark.py](osmo/ood_benchmark.py) | [Eight L40S GPUs](osmo/ood_baseline_l40s.yaml) |
| Recorded-condition solver gate | [runtime_probe.py](osmo/runtime_probe.py) | [One L40S](osmo/runtime_probe_l40s.yaml) |
| Paired OOD controls and reversal | [ood_steering.py](osmo/ood_steering.py) | [Eight L40S GPUs](osmo/ood_steering_l40s.yaml) |
| Distributed OOD shards | [ood_distributed.py](osmo/ood_distributed.py) | [Eight independent one-L40S tasks](osmo/ood_distributed_l40s.yaml) |
| Genuine development integration smoke | [astra_dev_smoke.py](osmo/astra_dev_smoke.py) | [One L40S](osmo/astra_dev_smoke_l40s.yaml) |
| Recorded genuine-proposal diagnostic | [astra_proposal_replay.py](osmo/astra_proposal_replay.py) | [One L40S](osmo/astra_proposal_replay_l40s.yaml) |

The distributed workflow changes scheduling while preserving the logical shards
and resolved method configurations. Each worker repeats the required gates.
The development smoke uses one standard task-0/state-0 episode, seed 7, and the
full 520-action preset: validation does not permit an 80-action benchmark budget.
It checks several genuine proposals, advancing observations, and exact latent
reuse; it is an interface check without a benchmark success-rate claim.

[bootstrap.sh](osmo/bootstrap.sh) verifies the uploaded payload hash, creates and
activates its `emimic` environment, installs the isolated dependencies, and checks
the allocated GPU. [package_payload.py](osmo/package_payload.py) stages authorized
local inputs; completed runs use their frozen payload hashes. Checkpoint weights
are downloaded and verified on the worker. Reports and videos are periodically
archived under `s3://rldb/experiments/astra-reversal-20260924/<workflow-id>/`;
the final archive contains lossless arrays.

New payloads include `tests/unit/astra`, the native integration test, and its
small fixture. The standalone development entrypoint runs these tests without
the repository-wide pytest hooks. A freshly extracted payload passed all 161
Astra unit tests and all seven native integration tests. Existing experiment
payloads remain unchanged and retain their recorded source hashes.

The recorded-proposal diagnostic fixes development steps 310, 330, and 485
(one passing control and both failures) and RK4 resolutions 100, 200, and 500.
Its [input plan](reports/astra_development_replay_plan.json) predates the GPU
replay. It verifies exact original arrays, conditions, and provider responses;
measures reconstruction before clipping; and reports N100 reproduction
separately. It makes no new Astra calls, uses no OOD observations, and selects
no solver. The [completed replay](reports/astra_development_replay.json) verifies
all original N100 latents and endpoints bit-for-bit. Step 330 still exceeds 0.02
at every tested resolution; step 485 passes at N200 and N500. These three
deliberately selected endpoints do not establish general reconstruction accuracy.
Use the packager's
`--include-astra-proposal-replay --output NEW_PATH/payload.tar.gz` flags to
include the verified input bundle without replacing a frozen upload.

The checked-in [OpenPI-input reversal JSON](configs/pi05_libero_openpi_inputs_reversal.json)
is the historical development template with RK4/50 and an unset agent model.
Current OOD and development-smoke drivers resolve and freeze RK4/100, CUDA,
the actual Astra model, and the settings above before execution. The template
alone is not the current experiment configuration.

## Local checks and records

Activate the repository's `emimic` environment before project Python tooling.
In the PR checkout, run these commands from the repository root:

```bash
python -m pytest tests/unit/astra -q
python -m astra_reversal preflight
python -m astra_reversal inspect-checkpoint astra_reversal/.deps/checkpoints/pi05_libero_base
```

The optional native-policy integration test uses the installed policy dependencies
and can read authorized local input assets through `EGOVERSE_TEST_PI05_INPUT_ASSETS`:

```bash
python -m pytest --integration tests/integration/test_astra_lerobot_policy.py -q
```

Fresh-main validation passed **468 repository unit tests**, with five existing
optional OpenPI tests skipped. All **seven native LeRobot integration tests**
passed separately in the pinned runtime. The Astra-specific total is 161 unit
tests plus those seven integration tests. The standalone tests support both
the `pytest` console command and `python -m pytest`. Local tests do not substitute
for the recorded GPU gates or environment rollouts.

Runs save resolved configuration, source and checkpoint hashes, frozen manifests,
observations, prompts, full flow endpoints, executed actions, clipping,
completion/retry/fallback events, latency, and video. Large arrays are lossless
`.npy` files referenced by hash from JSONL. All episode state is cleared between
runs. The [read-only Stage 1 audit](audit_astra_inversions.py) validates those
references, pairs each Astra inversion with generation under the same condition
and latent, and separately checks reuse on subsequent observations:

```bash
python -m astra_reversal.audit_astra_inversions path/to/completed/reversal_run \
  --output path/to/runtime_inversion_audit.json
python -m astra_reversal summarize path/to/completed/run
python -m astra_reversal compare path/to/left/run path/to/right/run --samples 2000
```

Outputs under changed conditioning are not reconstruction-error measurements.
Paired evaluation requires matching task/state/seed keys, reset hashes, protocols,
budgets, and splits; technical failures remain in episode denominators. Physical
feasibility, wrong-object actions, repetition, and mode drift require separate
rollout review. The [SPEC](SPEC.md) retains the full Stage 1 and Stage 2 contracts
and the additional evaluation requirements.
