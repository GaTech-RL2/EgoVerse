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
These baseline implementations have been inspected and pinned, not run here.

Primary sources
RTC: https://arxiv.org/html/2506.07339v1 (equations 2–4)
DSRL authors' code: https://github.com/nakamotoo/dsrl_pi0
RLinf: https://github.com/RLinf/RLinf/tree/c70606f08cdca259b8dec03d4430926b5b8fac9d

Status
First implementation is present. Local checks: 38 new CPU tests passed; full
repository unit suite: 1700 passed, 6 optional skips. The new opt-in action-VJP
integration test passed in the pinned LeRobot runtime. These are synthetic/small
model checks, not full-checkpoint GPU or robot success results.

No new robot SR or sample-efficiency result exists yet. The user authorized a
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
