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

No new robot SR or sample-efficiency result exists yet. The MD's handoff requires
the authorized experiment budget before experiments. A budget question is pending;
the suggested 12 L40S GPU-hours has not been treated as authorization. The config
therefore has launch_allowed=false. Strong-baseline parity, full-weight GPU
preflight, data collection, learning curves and confirmation remain outstanding.

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
