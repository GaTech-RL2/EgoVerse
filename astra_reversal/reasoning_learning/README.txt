Reasoning-guided policy learning — implementation and study plan

Source: ../proposals/reasoning_guided_policy_learning_20261003.md
Protocol: ../configs/reasoning_policy_learning_v1.json
Base branch: astra/demo-skill-library-20260930, commit d4b2b690, PR 704.

Objective
Better autonomous success per new environment interaction than strong RL, from
the same pi0.5 checkpoint. Assisted success or CPU checks alone are insufficient.
FRS action steering is excluded. No action inversion is used in this package.

Plan
1. Pin the checkpoint, deployment interfaces, baseline sources, autonomous
   threshold, reset schedules, and compute budget before experiments.
2. Implement/test normal-direction guidance, fixed-reference candidate judgments,
   single execution, evidence admission, and actual policy parameter updates.
3. Run an OSMO L40S preflight with the full weights. Verify zero-guidance native
   parity, action gradients, zero-adapter parity, training gradients and memory.
4. Pilot on Goal OOD task 6 and Spatial OOD task 2, three seeds, eight collection
   rollouts per task/seed. Evaluate autonomously at 0/2/4/8 collection rollouts,
   on ten separate reset states. The preregistered threshold is 80% success.
5. Compare tuned DSRL and PPO using the same checkpoint, observations, tasks,
   budgets and evaluation schedule. Diagnose losses, revise on development data,
   and confirm a frozen recipe on all 20 tasks with fresh seeds/resets if budget
   permits. Report uncertainty, censored threshold crossings, costs and failures.

Implementation
LeRobot uses t=1 for noise, t=0 for actions. The estimated endpoint is x-t*v.
Guided native velocity is v + lambda(t)*grad_x E; negative integration time steps
make this gradient descent. The schedule strength*t, projection of direct edits
onto mask support, and gradient-norm clipping are explicit experimental choices.
This is RTC-inspired, not an exact reproduction of RTC's weighting. Coupling in
the model can still alter unselected endpoint components.

Astra (gpt-6-astra, medium, Codex CLI through the authenticated local relay) sees
both real cameras, state8, instruction, controller semantics, native/candidate
commands, and recent real feedback. Its roles are diagnose, compare and assess.
It cannot execute candidates or query privileged simulator object coordinates.
Each comparison freezes the observation, policy version, rule and native sample.
An exclusive durable claim permits only one execution. Rejected proposals count
as search, not correction episodes. Episodes, chunks and assisted simulated
seconds are counted separately. The simulator pauses during reasoning; this is
not a real-time controller. Initial autonomous evaluation is for the common
learning-curve schedule, not a required failure demonstration to trigger Astra.

Native flow-matching loss updates rank-8 LoRA parameters inside the action
transformer's attention/MLP layers. These are actual policy parameters, not the
previous external residual action head. Zero output adapters preserve the initial
policy; float32 adapters and optimizer state retain small updates. The VLM stays
frozen. Targets are fixed real commands, with fresh noise and native Beta times.
Original-demo replay beta and synthetic candidate training weight both start at 0.

Full ten-step executed action windows are paired with their own pre-action
observations. Every step must have observed-useful evidence, and correction steps
also need a pre-execution predicted win. Include useful setup and continuation;
exclude ambiguous/failed segments and unexecuted tails. Low admission is a failure
mode to measure before adding synthetic targets or partial-label objectives.

Baseline audit
Official RLinf revision c70606f08cdca259b8dec03d4430926b5b8fac9d supplies:
  examples/embodiment/config/libero_spatial_dsrl_openpi_pi05.yaml
  examples/embodiment/config/libero_spatial_ppo_openpi_pi05.yaml
The example checkpoint RLinf/RLinf-Pi05-LIBERO-SFT differs from the selected
lerobot/pi05_libero_base. Never silently substitute it. RLinf's OpenPI loader
does not establish parity with the LeRobot export. Key conversion, strict tensor
loading, processor/velocity/action parity and OOD reset integration remain to do.
The implementations were initially inspected and pinned; the current results below supersede that initial status.

Primary sources
RTC: https://arxiv.org/html/2506.07339v1 (equations 2–4)
DSRL authors' code: https://github.com/nakamotoo/dsrl_pi0
RLinf: https://github.com/RLinf/RLinf/tree/c70606f08cdca259b8dec03d4430926b5b8fac9d

Historical status at initial implementation
First implementation was present. Local checks: 38 new CPU tests passed; full
repository unit suite: 1700 passed, 6 optional skips. The new opt-in action-VJP
integration test passed in the pinned LeRobot runtime. These are synthetic/small
model checks, not full-checkpoint GPU or robot success results.

At that initial checkpoint, no new robot SR result existed. The user authorized a
total of 24 L40S GPU-hours on 2026-10-03, including setup, failures and all methods.
The config now has launch_allowed=true. Strong-baseline parity, data collection, learning curves and confirmation remain outstanding.

Launching after the budget is resolved
Use the existing immutable payload/source_identity.json/bootstrap.sh bundle
format, with ASTRA_ENTRY_MODULE=astra_reversal.osmo.reasoning_policy_learning.
Prepare one time-limited L40S worker:
  python -m astra_reversal.osmo.prepare_learning_launch BUNDLE OUTPUT \
      --phase preflight --gpu-hours HOURS
The preparer verifies committed source and bundle hashes; it does not submit.
Use phase=pilot only after weighted preflight succeeds. A pilot worker handles
one task and seed. Sum all worker allocations against the study budget. OSMO task
start/end times count bootstrap cost; the internal cost file labels its exclusion.

2026-10-03 full-weight preflight
One OSMO L40S completed the native parity and gradient gates. Zero guidance and
zero adapters both have max absolute error 0. Input gradient norm 0.01508;
adapter-loss gradient norm 0.0009536; peak allocation 14.92 GB; guided sampling
0.476 seconds. Total allocation including bootstrap: 0.1586 GPU-hours. See
full_checkpoint_preflight.json. No environment actions or optimizer updates
were performed by this diagnostic. Strength-1 guidance reduced this particular
probe target error by only 0.076%, so an effect-size sweep is the next check.

RLinf compatibility audit
The official converter maps 812 source tensors to all 667 expected tensors with
matching shapes. Weighted parity is pending. The standalone core loader bypasses
Ray/robot factory initializers but does not alter the pinned math modules. The
audit compares unmodified RLinf, then explicitly changes Gemma and SigLIP GELU
approximations to match the source checkpoint, restoring them afterward. Shared
native preprocessing avoids attributing processor changes to checkpoint errors.
Neither shape parity nor activation harmonization alone qualifies an RL result.

2026-10-03 RLinf weighted parity
Activation matching reduced maximum velocity error from 0.00268 to 2.39e-6,
and maximum normalized/controller action error from 0.00162 to 4.18e-7.
This passes the declared tolerances on one real observation and two noise draws;
it is not exhaustive task equivalence. Source weights load strictly. The explicit
activation compatibility change is necessary before PPO uses this checkpoint.

DSRL integration
The new serial harness reuses RLinf's released DSRL actor, encoders and ten-Q-head
network/methods directly, with the exact frozen native LeRobot decoder. It uses
SAC with mean Q aggregation, gamma=.999 per control step, tau=.005, no backup
entropy, softplus temperature initialized to 1 with target entropy -16, actor
lr1e-4, critic/temperature lr3e-4, 200 updates per collected rollout, and a
10-update critic warmup. Float32 master parameters use bf16 autocast. Serial
collection, batch64, and duration-discounted -1+success over each actual prefix
are explicit integration choices. No demonstrations are supplied to the baseline.
The 300-action finite horizon is terminal. DSRL's released noise actor repeats
one 32-D tanh-Gaussian sample across the horizon; evaluation uses its mean.
Consequently its pre-update policy has the same decoder weights but a different
noise distribution from the native sampler. Report both initial scores; do not
call these identical initial action distributions. OOD resets/evaluation cadence
and counted physical control steps match the teacher study.

Guidance revision under test
The v1 sweep (guidance_v1_effect_size.json) reduced a normalized +0.1 z target
error by only 0.019%, 0.076%, 0.379%, 0.758% at strengths .25/1/5/10. This is
insufficient evidence of a usable intervention. Two implementation choices are
now separate ablations: (1) tapering the coefficient to zero near the action
endpoint, and (2) projecting the already-masked loss gradient onto the same
latent coordinates, which discards coupling directions. The optional RTC
coefficient is min(100, strength*((1-t)^2+t^2)/(t*(1-t))) in native noise=1 time;
the full VJP option masks the endpoint error without a second latent projection.
The original settings remain defaults for reproducibility. A zero-action GPU
probe will compare these choices before changing the collection recipe.
Source: https://arxiv.org/html/2506.07339v1#S4.SS1, equations 2--4. Binary masks
make our squared-mask energy match RTC's weighted correction; soft masks differ.
No FRS, physical candidate retry, or unexecuted synthetic training is added.

PPO integration (GPU backward/recompute check now passed)
The second baseline uses RLinf's released Pi0RL stochastic flow sampler and
log-probability recomputation on the strict converted checkpoint. Its flow-SDE
noise level is .5, with one selected stochastic denoising step and native ODE
elsewhere, as in the released configuration. The full action expert (including
AdaRMS and action/time projections) and the upstream VLM-pooled value head train;
the VLM stays frozen. Serial PPO uses chunk-summed log probabilities, GAE .99/.95,
clip .2, value clip .2, Huber delta10, AdamW actor lr5e-6/value lr1e-4, betas .9/.95,
weight decay .01, grad clip1, one epoch per rollout, microbatch1 accumulated over
8 transitions. These serial batch/horizon/float32 choices are explicit overrides
of the released distributed configuration. Only the executed prefix enters its
log-probability objective. Full action-expert parameters are saved at evaluation
milestones; these deployment checkpoints omit optimizer state. Each GPU job must
pass initial native parity and rollout-versus-recompute log-probability agreement
and backward checks before collecting any environment data.

2026-10-03 guidance probe and collection v2
The 13-configuration zero-action probe completed in 0.1059 L40S GPU-hours
including setup (workflow astra-pi05-reasoning-learning-20261003-guidance-probe-1).
At strength10, RTC weighting with the extra latent projection reduced the masked
target error 68.10%, versus 0.758% for the original taper. Removing the projection
raised this to 71.51%, but the maximum unmasked output change grew from 0.000071
to 0.08544. These are diagnostics on one archived real observation, not robot SR.
Full records: guidance_v2_effect_size.json.

The v2 protocol retains v1 source/results and offers three computational candidates:
RTC strength5 projected, RTC strength10 projected, RTC strength10 full gradient.
All share the fixed native proposal's observation/noise and still need Astra's
clear predicted win before any execution. Bounds and evidence gates are unchanged.
Diagnosis now receives the preceding actual monitor images and the current images,
short observed-outcome evidence, and summaries of up to three previous collection
attempts. Autonomous evaluation results are never supplied to Astra. The teacher
system prompt/schema is unchanged; the additional context is explicitly labeled.
Use prepare_learning_launch --protocol-version v2 --phase pilot to select this
recipe. A /tmp/astra-stop-after-rollout marker ends new workers only after the
current collection and update are saved, with an explicit completion status.

The initial v1 Goal OOD6 / seed173 autonomous evaluation scored 5/10. During its
first collection Astra rejected very weak lift candidates as ties. This motivates
the weighting revision; it is not evidence that learning has improved autonomy.
At the v2 launch, DSRL was running and PPO still required its GPU preflight. Current measurements are below. Research objective unmet.

Prospective v3: express a gripper correction
The saved v1 trajectory reveals a second limitation. At step175 Astra tried
three overlapping +0.5 gripper edits, which the cumulative bound rejected. At
step210 it explicitly reported that +0.5 could not change the native opening
commands (approximately -1) into closing. The v3 protocol keeps six motion
channels at a maximum additive change of 0.5, but permits up to 2 on the gripper.
Final teacher targets must still satisfy exactly the original controller bounds;
invalid targets are rejected, not clipped. This makes an ordinary grasp/hold
request expressible without increasing motion bounds. Its rollout effect remains
untested. The system prompt and all v1/v2 response schemas remain unchanged.

V3 autonomous evaluation records videos, every actual command/reset audit and
the initial observation, avoiding redundant per-step image uploads when no
training data is collected. Collection still retains all pre-action observations.
This affects artifact I/O only, not observations supplied to the policy or actions.

Shared native evaluation and reporting
Repeated revisions of the same unmodified native policy can reuse the measured
Goal6/seed173 initial evaluation. The guard requires matching weight, tokenizer,
processor, model/adapter source, runtime, protocol, task, seed, reset-state,
reset-model and BDDL identities, plus exactly zero adapter/guidance parity error.
It records the original workflow and checksum and counts zero new interactions
or independent evaluation replicates. Updated policy checkpoints always run fresh
autonomous evaluation. Other task/seed combinations still run their initial
evaluation. A requested early stop now finishes any due scheduled evaluation
before starting another collection rollout.

study_report.py builds a local HTML dashboard with a method diagram, completed
autonomous curves, Wilson intervals, token accounting, retained partial samples,
videos, the teacher prompt, CSV and JSON. Missing evaluations never become zero
success. Local completed CLI calls count even if the worker was cancelled before
receiving the response. The report deliberately does not declare the research
objective met; that requires a separate supported comparative conclusion.

2026-10-04 00:25 UTC development evidence
DSRL completed Goal OOD6 / seed173: autonomous success 4/10 initially, then
5/10 at 2, 4 and 8 collected rollouts. Final collection cost 2,057 controls;
evaluation cost 8,455 controls; eight policy updates; two collection successes.
Total OSMO allocation including bootstrap was 0.8088 L40S GPU-hours. Its final
score ties the separately measured native Gaussian baseline (5/10). The 8/10
threshold was not reached. Full per-reset summary: dsrl_goal6_seed173_result.json.
This is one development task/seed and one serial RLinf-based configuration,
not a tuned multi-seed state-of-the-art reproduction or a superiority claim.

The v2 teacher's one diagnostic rollout failed after 250 actions plus ten
stabilization controls, using three correction episodes, eight assisted chunks
(two simulator seconds), and 80 Astra calls. CLI usage: 1,493,455 total tokens,
including 772,096 cached input tokens. Nine full windows were admitted: eight
native setup/continuation windows and one correction window at steps 150--160.
One 20-step native LoRA update was saved, but no updated autonomous evaluation
was due before the planned stop. Its learned SR is unknown, not 5/10 or zero.
Actual allocation: 0.7491 GPU-hours. See pilot_v2_failure_audit.json.

PPO passed native weighted parity (controller error <=4.18e-7), exact rollout
versus recomputed log probabilities, and nonzero actor backward gradients.
It has completed two collection rollouts and its first updated autonomous score:
5/10 initially and 5/10 after 620 collection controls. The full eight-rollout
experiment is still running; these interim numbers are not its final result.

The first v3 worker passed the exact native-reference reuse checks but its local
tunnel started before OSMO marked the worker ready (HTTP 425). No Astra response
arrived; the worker timed out after initialization, consuming 0.1471 GPU-hours.
The explicit infrastructure retry keeps the same immutable worker payload and
protocol. The launcher now uploads first, then opens the tunnel, and terminates
its local relay if that tunnel exits. The retry is receiving real Astra calls.
No failed startup is silently removed from the compute/interaction accounting.

Standalone PNG/PDF plots now accompany the interactive offline dashboard.
Latest updates without a subsequent scheduled autonomous evaluation are labeled
unevaluated, so initial native scores cannot be mistaken for learned outcomes.
The study remains in development; repeated seeds, Spatial OOD2, baseline tuning
and any fresh confirmation remain outstanding within the authorized budget.
