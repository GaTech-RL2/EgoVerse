# Astra Reversal: method and implementation plan for LIBERO-Long and LIBERO-OOD

## 1. Research objective

In Stage 1, use Astra as a slow agent that **directly generates both a high-level intent
and a continuous robot action trajectory**. Invert Astra's trajectory through
a frozen pretrained flow policy to recover a noise tensor, then feed that noise
back into the policy to generate actions from fresh observations at a higher
frequency.

Astra's physical priors motivate direct numeric action generation. How well
those priors transfer to LIBERO's controller conventions is an experimental
question. The implementation should measure that ability directly.

Stage 2 continues these experiments with **Astra augmenting observations and
language**, while reference actions come from a declared source. The reference
actions are inverted through the frozen pretrained **π₀.₅** flow head; recovered
noise and augmented conditioning then drive its updated forward pass. Astra need
not emit continuous actions in Stage 2. Sections 2–10 specify Stage 1; Section 11
defines the Stage 2 extension and the contracts that change.

This is the single project specification: it contains the architecture, agent
contract, equations, control loop, configuration, evaluation, and implementation
plan. It replaces the previous grounding-module design, classifier sketch, and
preliminary configuration/schema files. Status: design only; no policy or
simulator experiments have been run for this document.

Start with **LIBERO-10 (`libero_10`)**, commonly called LIBERO-Long, and a
LIBERO-compatible π₀.₅ checkpoint. The bundled OpenPI example reports 92.4%
success for its π₀.₅ checkpoint on this suite; this has not been reproduced here.
Standard-suite improvement and generalization to unseen compositions must be
evaluated separately. Add **LIBERO-OOD** from
[Li, arXiv:2505.03500](https://arxiv.org/abs/2505.03500v5) as the published
compositional-generalization benchmark, alongside LIBERO-Long. It contains
20 extrapolated tasks, split into two ten-task suites. This is a separate
benchmark from LIBERO-10, not an alias or replacement.
The previously mentioned reversal paper has not yet been
identified, so this plan makes no claim of novelty or fidelity to that paper.

## 2. End-to-end method

1. Give Astra the task instruction, current camera observations, robot state,
   recent progress, and the exact action-space specification.
2. Astra outputs the active subgoal, completion criterion, and a short numeric
   action chunk in LIBERO controller coordinates in the same response.
3. Validate the response and apply the pretrained policy's normalization and
   padding transforms to the numeric chunk.
4. Integrate the frozen flow policy from action space to noise space, using the
   current observation and task instruction as fixed conditioning.
5. Pass the recovered noise explicitly into the policy's forward sampler.
6. Execute the first few generated actions, observe again, and generate another
   chunk from the recovered noise with the new observation.
7. Ask Astra for a fresh intent/action proposal periodically or when progress,
   completion, or timeout requires a new decision.

Compact notation:

```text
(h_k, A_astra_k) = Astra(task_instruction, observation_k, history, action_spec)
A_model_k       = NormalizeAndPad(A_astra_k)
Z_k             = InvertFlow(A_model_k | observation_k, task_instruction)
A_policy_t      = SampleFlow(Z_k | observation_t, task_instruction)
A_env_t         = DecodePolicyActions(A_policy_t)
```

**Astra owns all numeric motion generation in the proposal.** There is no
separate grounding model, waypoint controller, inverse-kinematics stage, or
π₀.₅ proposal generator in the primary method. The adapter only validates and
transforms the numbers Astra supplies; it does not infer missing motions,
interpolate sparse waypoints, or silently repair an incomplete trajectory.

An action mode means a continuous behavior such as reaching around the right
side of an object. There is no required discrete skill classifier. The recovered
noise is the proposed mechanism for retaining Astra's motion intent while π₀.₅
responds to changing observations.

## 3. Astra input and output contract

### Inputs

Provide the full instruction, current agent-view and wrist images, end-effector
pose, gripper state, and a bounded history of actions, completed subgoals, and
failures. Attach episode and observation IDs to each request.

Also provide a versioned action specification containing:

- Required chunk length `H`, timestep duration, and seven channel names/order.
- Whether values are controller deltas or absolute commands.
- Translation/rotation frame, rotation representation, scaling, and bounds.
- Gripper open/close convention and permitted values.
- A statement that output is in environment controller-input coordinates,
  before checkpoint normalization and model padding.

Resolve these from the actual LIBERO controller and checkpoint. Do not describe
controller-scaled deltas as meters or radians unless that is the verified API.
Astra should receive the same coordinate convention used by the environment.
A bare list of target positions or a natural-language subgoal is insufficient:
Astra must supply all seven channels for each of the `H` timesteps.

The primary observation contract uses images and robot proprioception. No
simulator object poses or ground-truth success predicates are provided to Astra.
If extra depth or calibration inputs are later added, declare them and match
the information available to relevant baselines.

### Outputs

Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Version of the response contract. |
| `episode_id`, `plan_id`, `observation_step` | Provenance and response freshness. |
| `subgoal_id`, `subgoal_instruction` | Current intent, retained until completion or replanning. |
| `action_spec_id` | Exact controller convention used to generate the numbers. |
| `action_chunk` | Finite numeric array of shape `[H, 7]`, before checkpoint normalization. |
| `completion` | Supported completion-check type plus its parameters. |
| `timeout_env_steps` | Deadline for the active subgoal. |

For example, a response may say “approach the mug handle from the right with
the gripper open” and include ten rows of translation, rotation, and gripper
commands. Each row is an action for one environment timestep. It is a local
proposal for the next half-second at 20 Hz, not the entire multi-second task.

The initial prompt should ask Astra to maintain or revise its current subgoal
and regenerate its local action chunk from the latest observation. Returning
new actions does not automatically mean advancing to the next subgoal.

Validate the schema, shape, finite values, bounds, action specification, and
timestamps before inversion. Reject malformed proposals and allow one bounded
regeneration attempt; if it fails, use a declared π₀.₅ baseline fallback and log
the event. Do not silently pad missing action timesteps, clip a proposal into a
different motion, or ask another model to synthesize its missing values.

Keep the original response, accepted numeric array, and transformed tensor in
the record. Physical plausibility remains an evaluation outcome: satisfying
shape and controller bounds does not prove a motion is feasible.

### Completion checking

Map the agent's completion request to implemented checks, rather than treating
free text as executable logic. Start with checks based on observed gripper/robot
state and an image-based progress assessor where necessary. Report the assessor's
cost and errors. Simulator predicates are reserved for evaluation.

Check progress at each low-level refresh. Require stable completion evidence
before advancing, initially two consecutive positive checks. More expensive
Astra progress judgments happen at its scheduled refresh or on an explicit
event and count toward its call budget. Initial subgoal timeout is 60 steps
with at most two recovery replans per subgoal; tune on development data only.

## 4. Flow inversion and generation

Use the convention verified in the local OpenPI implementation:

```text
c = (observation, language)
x(t) = (1 - t) A + t Z

t = 0: normalized action endpoint
t = 1: noise endpoint
dx/dt = v_theta(x, t, c)

I_c(A): integrate 0 -> 1 to recover noise
F_c(Z): integrate 1 -> 0 to generate actions
```

An initial Euler implementation with `N` steps is:

```text
Inversion:
  x_0 = NormalizeAndPad(A_astra)
  x_(j+1) = x_j + (1/N) v_theta(x_j, j/N, c_inv)
  Z_hat = x_N

Generation:
  y_0 = Z_hat
  y_(j+1) = y_j - (1/N) v_theta(y_j, 1-j/N, c_exec)
  A_model = y_N
```

Use integer-indexed time grids. Keep the observation and prompt fixed within
each solve. The primary method uses the full task instruction for both inverse
and forward conditioning. Astra's subgoal guides its own numeric output and
the progress monitor; it is not automatically added to the policy prompt.
A subgoal-conditioned policy is a separate baseline/ablation.

The local `pi05_libero` configuration uses `H=10` and default internal action
dimension `D_model=32`; the environment receives seven channels. Read actual
dimensions from the loaded checkpoint. The inverse consumes `[B,H,D_model]`.

Apply the checkpoint's exact action conversion, normalization, and padding
order. Astra emits the seven physical control channels; the adapter supplies
the checkpoint-defined padding. Preserve all internal channels during inversion
and generation, and unnormalize/drop padding only at environment execution.
Do not apply action normalization to the recovered noise. Log any output
clipping required by the environment adapter, and use identical rules for baselines.

### What the round trip means

For an ideal deterministic ODE under regularity assumptions,
`F_c(I_c(A)) = A`. Discrete Euler integration only approximates this identity;
reversing Euler steps is not an exact inverse of the discrete sampler.

Consequently, with identical conditioning, the first decoded action chunk
approximately reproduces Astra's proposal. Reversal does not automatically
improve a proposal or project it onto physically feasible motions.

The proposed value appears at subsequent policy calls:

```text
Z_k = I_(o_k, task_instruction)(NormalizeAndPad(A_astra_k))
A_t = F_(o_t, task_instruction)(Z_k),  t >= k
```

The noise remains temporarily fixed, while observation `o_t` changes. The
hypothesis is that this retains Astra's action mode while the pretrained policy
adapts continuous control. It can also repeat a motion or drift away from the
intent; latent reuse must have a bounded lifetime and be evaluated.

A recovered noise tensor is neither a random seed nor a guaranteed semantic
code. Arbitrary Astra proposals may invert to non-Gaussian latents. Log norms
and distribution statistics without treating them as feasibility certificates.

Test solver resolutions `N=10,20,50`, then a higher-order solver if necessary.
Measure full internal and seven-channel action reconstruction, known-noise
recovery on policy-generated diagnostic samples, velocity evaluations, and latency.
Policy-generated samples are numerical tests, not the proposal source at runtime.

Optional later ablations:

- Noise mixing: `Z_mix = sqrt(1-rho^2) Z_hat + rho epsilon`. Zero mixing is
  pure reuse; `rho=1` is independent Gaussian noise. Intermediate mixtures
  need not be Gaussian.
- Partial inversion to `t=tau<1`, followed by generation from that same time.
  This needs an arbitrary-start-time sampler; never pass a partial state to a
  sampler assuming `t=1`.
- Changing the policy prompt to include the subgoal, evaluated separately so
  language conditioning and inversion effects remain distinguishable.

## 5. Control rates and scheduling

| Component | Initial schedule | Operation |
| --- | --- | --- |
| Environment | 20 Hz | Execute one seven-dimensional command. |
| π₀.₅ | Every 5 environment steps, nominally 4 Hz | Generate 10 actions; execute the first 5. |
| Astra + inversion | Every 20 steps, nominally 1 Hz, or on an event | Generate a new intent/action chunk and recover a new noise tensor. |
| Progress monitor | Every policy refresh | Check completion, stalled progress, timeout, and freshness. |

Sweep Astra periods of `10,20,40` steps. Model horizon, executed action count,
and agent refresh period are separate settings.

Astra's ten-action proposal covers 0.5 simulated seconds, while a 20-step
refresh period spans one second. During that interval, π₀.₅ produces four
fresh chunks from the same latent and updated observations. The system does
not loop or stretch Astra's original ten actions. Reusing a chunk-shaped latent
across these fresh horizons is itself an experimental assumption; compare it
with reused random noise and more frequent Astra updates.

Start synchronously, pausing simulation at chunk boundaries during agent calls.
The nominal frequencies describe simulated time, not demonstrated wall-clock
throughput. Record actual agent and policy latency.

A later asynchronous version must version requests by episode, subgoal, plan,
and observation. Accept results only at chunk boundaries, discard obsolete or
over-age results, and clear unexecuted actions when a plan is superseded. Continue
the current valid plan only until its deadline, then use the declared baseline
fallback. Record rejection, retry, and fallback costs.

Cache policy image/language prefixes only for an unchanged condition. Rebuild
them whenever observations or prompts change. Reset all queues, latents, and
history between episodes. Subgoal retries never reset the episode action budget.

### Control loop

```text
reset to prescribed LIBERO initial state; perform stabilization steps
clear plan, latent, history, and action queue

while action budget remains and environment success is false:
    if action queue is empty:
        o = capture current images and robot state
        progress = monitor(o, active_subgoal, history)

        if no valid plan or agent_period_elapsed or progress.requires_replan:
            proposal = Astra(task_instruction, o, history, action_spec)
            validate proposal; resolve completion checker
            # Rejected responses: bounded regeneration, then logged fallback.
            A_model = normalize_and_pad(proposal.action_chunk)
            Z = invert(A_model, condition(o, task_instruction))
            activate proposal and Z with timestamps and bounded lifetime

        A_model = sample(condition(o, task_instruction), noise=Z)
        A_env = decode_with_checkpoint_transforms(A_model)
        validate output; enqueue first 5 actions or use logged baseline fallback

    execute one action; record observations, action, latency, and success
    stop immediately if environment success is true
```

## 6. Unified configuration

The following is the initial design configuration, embedded here as the single
source of truth. It is not an existing runnable command or validated experiment.

```yaml
project: astra-reversal
seed: 0
benchmark:
  suite: libero_10
  control_frequency_hz: 20
  task_action_budget: 520
  stabilization_steps: 10
  trials_per_task: 50
  task_manifest: null  # Resolve and version prescribed task/initial-state IDs.
agent:
  backend: astra
  model_version: null  # Select and record the actual API/model.
  output: intent_and_continuous_action_chunk
  action_space: libero_controller_input_before_checkpoint_normalization
  action_spec_id: null  # Resolve from actual controller/checkpoint.
  refresh_env_steps: 20
  invalid_response_retries: 1
  subgoal_timeout_env_steps: 60
  subgoal_recovery_replans: 2
policy:
  config_name: pi05_libero
  checkpoint: null  # Resolve artifact, backend, transforms, and provenance.
  frozen: true
  prompt: full_task_instruction
  horizon_source: checkpoint  # Expected H=10 in the inspected configuration.
  model_action_dim_source: checkpoint  # Expected 32 internally, 7 for LIBERO.
  execute_steps: 5
flow:
  integrator: euler
  inversion_steps: 10
  generation_steps: 10
  inversion_time: [0.0, 1.0]
  generation_time: [1.0, 0.0]
  noise_mix_rho: 0.0
controller:
  synchronous: true
  latent_max_age_env_steps: 20  # Couple to refresh period in schedule sweeps.
  completion_positive_checks: 2
  fallback: full_instruction_policy_with_fresh_noise
evaluation:
  track: standard_libero_10
  save_planner_responses: true
  save_flow_traces: true
  save_rollout_videos: true
```

Freeze development and test manifests before tuning these values. Longer action
budgets or custom tasks require separately labeled configurations.

Benchmark presets for this embedded configuration:

| Preset | Suite IDs | Task action budget | Stabilization | Initial trials/task |
| --- | --- | --- | --- | --- |
| Long-horizon execution | `libero_10` | 520 | 10 | 50 |
| Published OOD composition | `libero_goal_ood`, `libero_spatial_ood` | 300 per suite | 10 | 10 |

Run the OOD suites separately and aggregate their 20 task results afterward.
Override the suite, budget, trial count, evaluation track, and environment
revision together; do not carry the LIBERO-10 defaults into the OOD protocol.
Section 8 specifies the source revision and success-predicate differences.

## 7. Planned implementation and records

All modules below are planned, not implemented. Keep implementation local to
`astra_reversal/` and reuse OpenPI's LIBERO conventions.

| Module | Responsibility |
| --- | --- |
| `agent.py` | Astra input formatting, direct intent/action generation, response validation, and replay of recorded Astra responses for debugging. |
| `action_adapter.py` | Controller specification, shape/bounds validation, checkpoint normalization/padding, and output decoding. No motion synthesis. |
| `policy_adapter.py` | Frozen checkpoint loading, shared condition preparation, velocity access, and explicit-noise sampling. |
| `flow.py` | Directional integration, schedules, inversion results, and reconstruction diagnostics. |
| `controller.py` | Two-rate loop, progress checks, latent lifetime, queue management, retries, and fallback accounting. |
| `libero_runner.py` | Task registry, prescribed initial states, observations, environment execution, terminal success, and video. |
| `evaluate.py` | Paired comparisons, uncertainty, per-task metrics, latency/compute accounting, and failure analysis. |

Records to implement:

- **AgentProposal:** the complete response contract from Section 3, raw response,
  agent model/version, prompt template, sampling settings, and request/response times.
- **InversionResult:** accepted controller actions, transformed full tensor,
  inverse observation/prompt IDs, solver/grid, checkpoint/normalization versions,
  recovered noise, and optional reconstruction errors.
- **ControlStep:** task and initial-state IDs, episode seed, active plan/latent ID,
  current observation/prompt ID, generated and executed actions, progress state,
  timestamps, retries/fallback reasons, and terminal success.
- **RunManifest:** repository versions, environment/controller/action specification,
  checkpoint data provenance, resolved configuration, and evaluation split.

The local PyTorch model already exposes `sample_actions(..., noise=...)` and
`denoise_step(...)`; it needs a public inversion interface and shared condition
preparation. Choose the backend from the actual checkpoint artifact. If converting
JAX weights to PyTorch, validate numerical parity before evaluating reversal.
A named checkpoint path does not mean the artifact is locally available.

## 8. Evaluation and ablations

### Standard LIBERO-10

Use all ten tasks and prescribed initial states. Match the local OpenPI protocol:
520 task actions plus ten stabilization steps, initially 50 trials per task.
Pair methods by task/initial state and repeat stochastic methods with independent
seeds. Cache identical requests for reproducibility, but regenerate agent output
when observations diverge between methods.

| Method | What it tests |
| --- | --- |
| Full-task π₀.₅ with fresh Gaussian noise | Baseline policy performance. |
| Full-task π₀.₅ with reused random noise | Whether any fixed latent improves temporal consistency. |
| Astra subgoal text fed to π₀.₅ with fresh noise | Whether hierarchical language alone explains gains. |
| Direct Astra continuous-action execution | The quality of Astra's own physical predictions. |
| Same-condition inversion/replay | Numerical correctness; not an improvement claim. |
| Astra continuous actions + inversion + fresh-observation decoding | The proposed method. |
| Proposed method with recovered latents replaced by random noise | Whether the recovered latent specifically matters. |

For the direct-Astra comparison, match both methods at an agent period of
`H=10` steps: direct execution consumes Astra's ten actions, while reversal
uses the same ten-step proposal window with policy refreshes every five steps.
In the proposed default 20-step schedule, direct Astra would otherwise run out
of actions after ten steps. Do not invent extra actions, repeat the chunk, or
claim matched call budgets. If direct Astra instead refreshes more frequently,
report its additional calls explicitly. At a common observation, replay both
decoders against the same Astra proposal; across diverging rollouts, each agent
must receive its own current observation.

Report per-task and macro-average full-episode success, paired differences and
uncertainty, actions/time to success, subgoal completion, invalid-response rate,
retries, fallbacks, mode drift/repetition, and failure categories. Environment
terminal success is authoritative; progress predictions are diagnostics.

Include actual wall-clock latency, agent calls, and flow velocity evaluations.
Add a compute-matched policy-sampling baseline with a specified non-oracle
selection rule. Compute paired bootstrap intervals over initial states within
tasks, accounting for repeated seeds per state. Tune only on development data.

Analyze Astra proposal quality separately from inverse accuracy and control
success: correct dimensions, controller-bound violations, numerical round-trip
errors, plausible motion, and eventual task completion answer different questions.

### LIBERO-OOD: published compositional-generalization benchmark

Add the benchmark introduced in **“VLAs are Confined yet Capable of Generalizing
to Novel Instructions”**, Quanyi Li,
[arXiv:2505.03500v5](https://arxiv.org/pdf/2505.03500v5), revised May 1, 2026.
Earlier versions/repository text use the title “Task Reconstruction and
Extrapolation for π₀ using Text Latent.” Source code is
[QuanyiLi/pi0-text-latent](https://github.com/QuanyiLi/pi0-text-latent);
the inspected repository revision is
[`587a6cbf64f1`](https://github.com/QuanyiLi/pi0-text-latent/tree/587a6cbf64f16c7b87fa5805dc0ed934192239a4).

The benchmark tests whether a policy can combine familiar grasping and placement
behaviors into an undemonstrated trajectory. For example, knowing how to put
cream cheese in a bowl and put a bowl on a cabinet does not necessarily yield
“put the cream cheese on the cabinet.” This directly tests the composition
objective of Astra + frozen π₀.₅.

| Suite | Tasks | Main challenge |
| --- | --- | --- |
| `libero_goal_ood` | 10 | New grasp/place combinations based on LIBERO-Goal, including changed layouts and object-transfer cases. |
| `libero_spatial_ood` | 10 | Transfer spatially specified grasp behaviors to new destinations, plus object-transfer placement cases. |

Paper Section 4/Figure 2 describes six task-composition cases and four additional
object/layout-transfer cases within each suite. Use the released BDDL task
definitions and registry as authoritative; do not reconstruct tasks from figure
captions. Familiar component motions do not establish that our particular
π₀.₅ checkpoint has the same training exposure as the paper's policy.

**Protocol.** The paper evaluates ten independent runs per task: 100 episodes
per suite, 200 across both suites. The inspected released runner uses ten trials
per task, a 300-action limit for both OOD suites, ten stabilization steps, and
five executed actions per policy query. Keep those limits for the initial
comparison and report each suite plus the 20-task macro-average. A larger
repeat count can be reported as an explicitly labeled extension.

Pin the benchmark environment, task assets, and success predicates, not only the
model. The repository README documents a contact-threshold change from 0.03 to
0.1 and accepts placement on top of the stove outside the narrower cooking
region. Label this the **released modified-LIBERO protocol**. Any strict,
unmodified-LIBERO predicate evaluation is a separate sensitivity analysis; do
not mix those outcomes in one score or compare them as identical protocols.

The released runner seeds the environment and resets it per trial, with the
standard task initial-state loading/application commented out. The paper
describes independent seeded runs. Before execution, lock and record the exact
reset/seed procedure used for reproduction, distinguish any per-episode-seed
extension, and pair methods using reproducible identical reset states. Do not
silently substitute LIBERO-10's prescribed-state protocol.

**Stage 2 experiment.** Run the full Section 11 matrix on both OOD suites using
the same frozen π₀.₅ checkpoint. Astra supplies observation/language augmentations;
the declared reference-action source supplies chunks for inversion. Measure
whether the augmented-condition inverse plus updated forward process can bridge
familiar components into the requested new combination. Report wrong-object,
wrong-destination, and demonstrated-location replay failures to distinguish
semantic composition from spatial memorization.

Include frozen π₀.₅, Astra subgoal prompting, augmentation-only, inversion-only,
combined Stage 2, and known-noise/reused-noise controls. Add explicit subgoal
prompt switching as a relevant comparison: the paper studies prompt switching
as well as text-latent interpolation. Its **text-latent interpolation (TLI)**
method is a literature comparator, not our action-flow inversion method. TLI
uses representations extracted from base-task demonstrations; disclose that
extra information if reproducing it. Published π₀/TLI scores are not measured
π₀.₅ baselines. Porting TLI to π₀.₅ is a separate optional implementation.

Apply the training-overlap audit below. Do not use OOD test demonstrations,
hand-tuned test-task phase schedules, or test outcomes to choose Astra prompts,
reference retrieval, augmentation rules, or solver settings. Preserve all task
IDs, reference sources, seeds, raw/augmented inputs, and environment revision in
the run manifest. Report suite/task success, paired uncertainty, latency,
reference-generation costs, and recovery outcomes under a common action budget.

**Integration plan.** Extend the runner's suite registry for the two published
OOD suites using the pinned task assets and environment changes, while keeping
the standard LIBERO-10 protocol selectable. Avoid overwriting the existing
LIBERO installation or shared path configuration during setup. First validate
all 20 task registrations, resets, and success checks; run a frozen-π₀.₅ smoke
test in each suite; then execute the matched Stage 2 comparison. Adding this
benchmark to the plan does not imply that its environment has been installed
or that the experiments have already run.

### Compositional-generalization validity and optional custom tasks

A checkpoint trained on the same LIBERO-10 tasks does not establish unseen-task
composition merely because Astra supplies new prompts. Audit training data,
demonstrations, retrieval, and planner examples. Mark unknown training overlap.

A primitive policy trained on LIBERO-90 may be useful, but suite membership alone
does not guarantee composition-disjoint testing. Define held-out combinations
of objects, relations, operations, and order, with known components permitted
and target combinations excluded from development/training.

If new LIBERO task specifications are necessary, validate feasibility and success
predicates and publish a versioned manifest. Label them custom LIBERO compositions,
not standard LIBERO-10. Evaluate success versus task length and unseen combinations
against the same baselines. Report all tasks and any development-selected hard
subset; do not select “unsolved” examples using final test outcomes.

## 9. Implementation plan and acceptance gates

### Phase 0 — Establish the environment and baseline

Pin LIBERO/OpenPI versions, controller action semantics, image orientation,
state layout, checkpoint backend, transforms, dimensions, and training provenance.
Create the executable configuration from Section 6 and the task/initial-state manifest.

Deliverables: verified action specification, frozen-policy loader, baseline video,
and measured baseline success. Gate: reference preprocessing and explicit-noise
actions match on identical input. Confirm `H` and both action dimensions before
writing the Astra prompt. Register the separate LIBERO-10 and LIBERO-OOD protocol
presets; pin the OOD fork and verify its 20 tasks and modified success rules.

### Phase 1 — Validate direct Astra action generation

Implement the joint intent/action response contract, including all `H × 7`
numbers. Give Astra observations and the verified controller specification.
Record genuine Astra responses for deterministic debugging; add invalid-response,
stale-response, and bounded-retry handling.

Deliverables: agent interface, response logs, proposal validity report, and a
direct-Astra development rollout. Gate: every accepted proposal contains finite
numeric commands in the declared space. No separate module synthesizes missing
motion. Measure physical performance instead of assuming it from prior capability.

### Phase 2 — Implement inverse and explicit-noise replay

Share condition preparation between the policy sampler and inverse. Implement
both integration directions, action encoding/decoding, and complete flow records.
Keep diagnostic policy-generated chunks separate from runtime Astra proposals.

Deliverables: flow integrator, policy/action adapters, reconstruction report.
Gate: constant/linear-field tests catch time/sign mistakes; known-noise model
round trips converge with finer integration within declared tolerances; the
forward adapter matches the reference sampler. Check transform, padding, and
conditioning mismatches deliberately. Choose tolerances from reference numerics
and controller sensitivity before main experiments.

### Phase 3 — Connect the two-rate controller

Implement the synchronous nominal 20/4/1 Hz loop, fresh-observation decoding,
bounded latent reuse, completion checks, queue invalidation, subgoal timeouts,
and episode resets. Add asynchronous operation after this behavior is verified.

Deliverables: controller/runner, annotated videos, timing logs. Gate: logs show
Astra supplying direct proposals less frequently than policy generation; tests
catch stale results, execution of superseded queued actions, and latent/history
leakage between episodes.

### Phase 4 — Run matched baselines and ablations

Debug on a development set, freeze settings, then evaluate the complete suite.
Compare direct Astra execution, subgoal prompting, plain policy, and reused
random noise. Sweep agent period and solver resolution; introduce optional
mixing or partial inversion only after full inversion is characterized.

Deliverables: paired rollout data, success/uncertainty tables, compute/latency
report, and failure clips. Gate: conclusions isolate the value of inversion from
Astra's numeric proposal quality, language planning, and extra computation.
A null or negative result is valid and should identify the failure mechanism.

### Phase 5 — Evaluate unseen compositions

Audit overlap, evaluate the published LIBERO-OOD suites, freeze any additional
custom composition manifests, and test task length and recovery
with matched observation and action budgets. Developing custom tasks or training
a suitable primitive checkpoint is a separate work package if needed.

Deliverables: split manifest, checkpoint provenance, composition results, and
ablations. Gate: generalization claims match the actual excluded combinations
and known training exposure.

## 10. First concrete experiment

On one preselected LIBERO-10 development task, obtain a direct Astra proposal
with ten continuous actions. Validate the inverse offline and verify identical-
condition reconstruction. Run paired control experiments for direct Astra,
subgoal-prompted π₀.₅, and inversion-guided π₀.₅ at a matched ten-step agent
period. Then test 20-step Astra refresh with five-step policy refresh to measure
the benefit or failure of slower agent calls and noise reuse.

The desired result is improved full-episode success or recovery under changing
observations at a measured compute cost. Noise reconstruction alone establishes
the numerical interface, not compositional generalization.

## 11. Continued Stage 2 experiments: Astra conditioning + reference-action inversion

### Objective and roles

Test whether Astra can guide a frozen pretrained **π₀.₅ flow policy** by
augmenting its observations and language, while a reference action trajectory
provides the action endpoint for inversion. π₀.₅ supplies both the reverse flow
and the updated forward flow, with shared, unchanged weights.

In this stage Astra outputs task-relevant visual annotations and a contextualized
instruction/subgoal. It does not have to generate continuous robot actions.
Replace Stage 1's mandatory agent `action_chunk` field with augmentation metadata;
record the reference trajectory separately with its source and coordinate space.

The pipeline is:

```text
current observation + original instruction + history
                    -> Astra
                    -> observation augmentation + language augmentation

reference actions + augmented observation/language
                    -> π₀.₅ reverse flow (action -> noise)
                    -> recovered noise

recovered noise + fresh augmented observation + augmented language
                    -> same π₀.₅ forward flow (noise -> action)
                    -> executable action chunk
```

Reference actions influence the forward pass through the recovered noise. They
are not a new independent input to the forward head. Adding an extra reference
action channel or a new embedding would require an architecture change and is
outside this frozen-policy experiment.

### Observation and language augmentation

Astra consumes the original images, proprioception, instruction, and recent
history. Its structured response can contain the active subgoal, relevant object
identities, image-coordinate boxes/points, a desired approach, task constraints,
and progress evidence. Preserve the original instruction in the assembled prompt
and append the current subgoal/constraints within π₀.₅'s token budget.

For observation augmentation, begin with deterministic rendering of Astra's
annotations on the existing camera images, such as a target outline or approach
marker. Keep an unmodified copy for logging and the agent's next observation.
Preserve camera ordering, image shape, and the checkpoint's preprocessing; keep
proprioception unchanged. Do not silently add image slots or arbitrary feature
vectors to an unmodified pretrained model. Crops or alternate renderings can be
separate ablations using supported input slots and explicitly recorded transforms.

The render step implements Astra's specified annotations; it does not synthesize
robot motion. Annotated images and new prompt formats may be outside π₀.₅'s
training distribution, so measure their effect against raw-image and language-only
controls. Discard or refresh stale visual annotations rather than copying old
pixel coordinates onto a changed scene without tracking. Between slow Astra
calls, a declared image tracker may update annotations from fresh frames; if
tracking is unavailable or loses confidence, expire the visual augmentation,
retain current language where valid, and record the change in conditioning.

### Reference-action sources

Start with one explicit source: a short action chunk sampled by the same frozen
π₀.₅ under the **raw observation and original task instruction**, before applying
Astra's augmentations. Preserve its full normalized internal tensor and seed.
This is an executable proposal, not future actions observed from the environment.

Other sources may be tested separately: aligned demonstration chunks for offline
diagnostics, permitted retrieval from a development/training reference library,
or cached policy proposals with explicit age/alignment checks. Demonstration
actions from the evaluation episode are privileged and cannot be used in the
main online benchmark. A previously executed chunk is not automatically a valid
future reference; its state and time alignment must be checked.

All references must match the checkpoint's horizon, embodiment, controller
conventions, and internal action dimensions. Normalize/pad external controller
actions once; do not normalize a policy's already normalized internal chunk again.
Record source observation/prompt, timestamps, transform version, and action space.

### Inverse and updated forward process

Let `l` be the original task instruction and `o_k` the observation at agent step
`k`. Let `m_k` be Astra's visual augmentation specification and `l_aug_k` the
assembled instruction. Define the primary experiment explicitly:

```text
(m_k, l_aug_k) = Astra(o_k, l, history)
o_aug_k       = ApplyObservationAugmentation(o_k, m_k)
c_raw_k       = (o_k, l)
c_aug_k       = (o_aug_k, l_aug_k)

A_ref_k       = F_pi05(Z_random | c_raw_k)       # Full normalized model tensor.
Z_hat_k       = I_pi05(A_ref_k | c_aug_k)        # Integrate t=0 -> t=1.

o_aug_t       = ApplyValidAugmentation(o_t, m_k) # Fresh pixels; track or expire.
c_aug_t       = (o_aug_t, l_aug_k)
A_out_t       = F_pi05(Z_hat_k | c_aug_t)        # Integrate t=1 -> t=0.
action_t      = Decode(A_out_t)[:execute_steps]
```

The inverse uses the reference actions as its initial flow state and Astra's
augmented observation/language as its condition. Recovering noise requires
integrating the flow velocity across time; the flow-matching training loss or
one head evaluation does not directly return that noise. Hold conditioning fixed
inside each solve and prepare a new observation/language cache whenever it changes.

At the refresh observation, replay under identical augmented conditioning
approximately reconstructs `A_ref_k`. Subsequent calls with fresh observations
test whether augmented conditioning and the recovered noise improve continuation
and recovery. This setup does not promise immediate correction of the reference
chunk. Inverting and decoding under identical conditions remains the numerical
control from Stage 1.

Also test **conditioning transfer** as a separately labeled variant: invert the
reference under `c_raw_k`, then generate under `c_aug_t`. This permits an immediate
conditioning-driven change in the decoded action. When the reference was sampled
from π₀.₅ under `c_raw_k`, its inverse should recover the known `Z_random`;
directly reusing that known noise is the required no-inversion control. Any gain
in that case must be attributed to augmentation/noise selection rather than
claiming that redundant inversion added information.

### Schedule and configuration changes

Inherit Stage 1's LIBERO budget, policy dimensions, normalization, and solver
conventions. Keep π₀.₅ frozen. Initially refresh Astra augmentations, the reference
chunk, and its recovered noise every 20 environment steps; decode with current
observations every five steps. Rebuild all three on a subgoal change. Include
reference-generation latency and velocity evaluations in compute accounting.

The conceptual configuration additions to Section 6 are:

```yaml
experiment_stage: 2
agent:
  output: observation_and_language_augmentation
  continuous_action_output_required: false
reference_actions:
  source: frozen_pi05_raw_conditioning
  retain_full_model_tensor: true
stage2:
  pretrained_flow_policy: pi05
  train_policy: false
  inversion_conditioning: astra_augmented
  generation_conditioning: fresh_astra_augmented
  observation_augmentation: annotations_on_existing_images
  language_augmentation: original_instruction_plus_subgoal_and_constraints
  invalid_visual_annotation: expire_and_log
```

These are overrides/extensions to the single embedded design configuration,
not an additional runnable config file. Reset augmentation state, reference
actions, and recovered noise between episodes.

### Experiments and implementation gates

| Experiment | Inversion condition | Forward condition | Purpose |
| --- | --- | --- | --- |
| Raw π₀.₅ | None; fresh noise | Raw observations and original language | Baseline. |
| Astra augmentation only | None; matched fresh noise | Augmented observations/language | Isolate augmentation without reversal. |
| Reference inversion without Astra | Raw | Fresh raw | Isolate inversion/noise reuse. |
| Stage 2 primary | Augmented | Fresh augmented | Combined reference inversion and augmentation. |
| Language-only / observation-only | Corresponding augmentation | Fresh corresponding augmentation | Separate the two conditioning contributions. |
| Conditioning transfer | Raw | Augmented | Test changing the conditioning between directions. |
| Known-noise transfer control | None; reference's known seed tensor | Augmented | Detect redundant inversion for policy-generated references. |

Match seeds, reference selection rules, task/initial states, execution budgets,
and refresh schedules. At identical observations, compare using the same saved
reference; across diverging rollouts regenerate references from each method's
current state. Include the reused-random-noise control from Stage 1. Report
reference quality, round-trip error, task success, recovery, stale annotations,
and total agent/policy computation separately.

1. **Augmentation interface:** implement and log raw/augmented observations and
   prompts; verify checkpoint input compatibility and annotation freshness.
2. **Reference interface:** implement full-tensor reference generation, provenance,
   and single-pass normalization for external references; validate dimensions.
3. **Conditional inversion:** expose independently selectable inverse and forward
   conditions; verify same-condition reconstruction and known-noise recovery.
4. **Online continuation:** connect the two-rate loop; test annotation expiration,
   subgoal refresh, cache invalidation, and episode-state resets.
5. **Matched evaluation:** run augmentation-only, inversion-only, combined, and
   transfer controls on LIBERO-10 and both LIBERO-OOD suites before any composition
   claim, keeping their budgets and success rules separate. Apply Section 8's training
   overlap audit and held-out-composition protocol to Stage 2 as well.

Record both conditioning paths explicitly in each inversion/execution trace.
The Stage 2 acceptance criterion is a measurable improvement in task continuation
or recovery beyond augmentation alone and reused noise, with its compute cost
reported. Accurate reconstruction alone does not establish that improvement.

## 12. References

These sources were inspected for the design. The references are implementation
evidence, not experimental results from this project.

- [LIBERO-OOD paper, pinned arXiv v5](https://arxiv.org/pdf/2505.03500v5)
- [LIBERO-OOD repository and documented predicate changes](https://github.com/QuanyiLi/pi0-text-latent/blob/587a6cbf64f16c7b87fa5805dc0ed934192239a4/README.md)
- [LIBERO-OOD runner: suite IDs, budgets, trials, and resets](https://github.com/QuanyiLi/pi0-text-latent/blob/587a6cbf64f16c7b87fa5805dc0ed934192239a4/examples/libero/main.py)
- [LIBERO example and upstream reported results](../external/openpi/examples/libero/README.md)
- [Runner: budgets, observations, and action queue](../external/openpi/examples/libero/main.py)
- [π₀.₅ LIBERO configuration](../external/openpi/src/openpi/training/config.py)
- [Model dimensions](../external/openpi/src/openpi/models/pi0_config.py)
- [PyTorch velocity and explicit-noise sampler](../external/openpi/src/openpi/models_pytorch/pi0_pytorch.py)
- [LIBERO action/observation adapter](../external/openpi/src/openpi/policies/libero_policy.py)
- [Checkpoint loading and transforms](../external/openpi/src/openpi/policies/policy_config.py)
- [Environment/controller wrapper](../external/openpi/third_party/libero/libero/libero/envs/env_wrapper.py)
- [Suite definitions](../external/openpi/third_party/libero/libero/libero/benchmark/__init__.py)
