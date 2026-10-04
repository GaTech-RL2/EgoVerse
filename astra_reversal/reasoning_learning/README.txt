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

Prospective v4: language subgoal candidates
The v2 videos/reviews show repeated lift attempts at the rack while the external
objective remains putting the bottle in the bowl. A version-gated extension lets
Astra choose either the existing bounded motor-target guidance or a concrete
current-phase subgoal instruction. The latter creates three computational
candidates with the same policy, real observation and native noise: full prompt
conditioning, or TEI between original/subgoal instructions at alpha .33 and .67.
There are no new demonstrations, altered images, simulated candidate outcomes,
physical retries or FRS. Every alternative still needs a predicted clear win
against the saved original-instruction native proposal. Only one prefix executes.
Executed useful windows retain their ORIGINAL task instruction for native LoRA
training and later autonomous inference. The subgoal is a teacher input only.

The extension changes the teacher schema/prompt only when explicitly enabled in
protocol v4. All 51 current v3 requests were compared with source 3439d378 and
retain identical schemas and complete payloads. Full-weight launch preflight will
exercise all three new conditioning paths and require exact native restoration
afterward. V4 is prepared for testing, not a demonstrated improvement. It keeps
the same eight-rollout collection and 0/2/4/8 autonomous evaluation schedule.

Baseline development tuning
The standard baseline setting is retained. A separately named more_reuse recipe
raises DSRL replay updates from 200 to 600 per collected rollout, with the same
batch64 and other SAC settings. PPO gathers two complete episodes from one fixed
policy version before an update, then trains for four epochs with optimizer
batch8. Terminal masks prevent GAE from leaking a following episode's reward
into the preceding episode. The original recipe uses one episode and one epoch.
Every scheduled evaluation still follows completed batches at 0/2/4/8 collected
episodes. This is additional development tuning, not a demonstrated advantage.

RL evaluations can use the same reduced artifact recording as teacher v3/v4:
all commands/reset audits and video, plus the initial raw observation. Collection
always retains every raw pre-action observation and exact PPO training records.
This changes artifact I/O only. Older immutable workers retain their original
recording behavior. All allocated wall time, including uploads, counts in the
compute ledger.

2026-10-04 completed standard PPO development run
Goal OOD6 / seed173 autonomous SR was 5/10 at collection0, 5/10 at2, 4/10 at4,
and 3/10 at8. The eight real collection rollouts used 1,864 controls including
stabilization, with three successes (resets4,6,7). Evaluation used 8,685 controls.
Eight policy updates completed; allocated runtime including bootstrap was
0.9251 L40S GPU-hours. The 8/10 threshold was not reached. This setting degraded
autonomous performance on the measured resets; it supplies no advantage claim.
See ppo_goal6_seed173_result.json and ppo_weighted_preflight.json. The development
tuning recipe was prepared before its first rollout and remains a separate run.

Prospective v5: tell the teacher what will actually execute
The driver has always executed five actions before replanning, but older teacher
requests exposed the ten-action proposal horizon without explicitly stating that
cutoff. Some saved comparisons penalized the unexecuted tail, and some target
edits constrained it. V5 states that only actions0--4 execute and restricts the
motor-edit schema to that prefix. Candidate judgments must still preserve necessary
subgoals, but an unexecuted bad tail alone is not evidence against the real prefix.
The complete ten-step training window requirement remains: those targets come
from two actually executed prefixes, never from a predicted tail. V1--V4 payloads
are preserved exactly. This revised interface is not yet a robot success result.

Training loss audit and v5 correction
The selected checkpoint declares output_features.action.shape=[7], while its
internal action dimension is32. The pinned LeRobot PI05Policy.forward truncates
per-dimension losses to7 before averaging (modeling_pi05.py lines1255--1257).
V1--V4 used the native flow-matching residual but averaged all32 channels. Those
runs therefore did not exactly match the LeRobot policy-level loss. Their results
and source are preserved as the padded-loss variant; they are not silently relabeled.
V5 excludes the25 padding outputs and adds a full-weight comparison between the
cached-prefix learner loss and LeRobot's direct training forward at identical noise,
time, observation and target. The native 32-dimensional sampling interface remains
unchanged. Whether the corrected objective improves autonomous SR remains untested.

V3 completed development result (OSMO pilot-4)
Two collection attempts failed after 620 total control steps, with seven assisted
five-action prefixes. Two LoRA updates trained 40 optimizer steps on 7 and 15
admitted windows. The updated autonomous policy scored 5/10, equal to native 5/10:
exactly the same five reset states succeeded, with no gained or regressed reset.
Teacher usage was 4,152,587 reported tokens (2,353,152 cached input), 216 calls,
and 2,681.80 seconds of CLI latency. Fresh evaluation used 1,952 controls.
Conservative allocation including worker initialization was 1.36369 L40S-hours.
The run stopped under a two-collection screening decision made before updated
evaluation outcomes were known. It used the historical 32-channel padded loss;
this is a measured tie, not a sample-efficiency advantage. Full evidence is in
pilot_v3_goal6_seed173_result.json.

Saved-data checks of verifier grounding
Four new comparisons on two archived V4 states tested explicit prefix timing
with and without the preceding real motion. Every selection remained native;
all twelve judgments were uncertain. The diagnostic executes no environment
actions and supplies no training labels. See comparison_history_probe.json.

visual_grounding.py introduces an OFFLINE feasibility check, not a deployed
intervention. Astra labels the visible grasp-center pixels in unaltered recorded
images without receiving robot poses. A separate local affine fit pairs those
labels with recorded XYZ proprioception and checks leave-one-out pixel error and
motion excitation. A good fit only measures consistency with VLM labels, not
independent camera accuracy or predicted action outcomes. No depth or object
poses are supplied. Any later live variant must derive its fit from its own
counted collection data, not import a calibration from these development runs.

Related primary sources: HAMSTER (https://arxiv.org/abs/2502.05485) uses coarse
2D paths with a trained downstream controller; RoboPoint
(https://arxiv.org/abs/2406.10721) trains image affordance prediction. Neither
establishes that unfinetuned Astra pixel labels will work here. 3D HAMSTER
(https://arxiv.org/abs/2606.31329) highlights the missing-depth problem in 2D
guidance. A projected point alone is therefore not a valid 3D placement target.

Deployment scope: the published checkpoint config specifies chunk_size=50 and
n_action_steps=10. This development study consistently generates ten actions
and executes five before replanning. Native/zero-guidance parity and all paired
comparisons refer to this shared runtime, not the publisher's stock rollout
settings. Both the runtime override and the serial RL integration must be
considered before claiming a strong published-baseline reproduction.

V6 prospective revision: TLI and computational rejection feedback
V4 and early V5 semantic proposals often differed too little or had uncertain
destination effects. V6 adds two TLI candidates alongside prompt and TEI: keep
the original task input and add 0.5 or 1.0 times the current-observation subgoal
text latent minus the original-instruction latent, after VLM blocks 0–16. Both
source banks are freshly captured from the same actual observation; no training
demonstrations are supplied. Fixed noise, frozen policy/rule/reference and the
clear-win gate remain in force. Zero-factor TLI and native restoration must
match exactly before collection.

Diagnosis previously received real outcome reviews but not the verifier's
reasons for declining its proposals. V6 includes the last three comparison
records, explicitly identified as past computational judgments. They may
inform the next diagnosis but cannot authorize its new candidate batch.
V1–V5 prompts and candidate pools remain unchanged.

The separate visible-pixel geometry probe produced only 3/8 labels above its
predeclared 0.6 confidence threshold. The minimum was six, so no projection
was fitted and no geometry intervention was deployed. The retained negative
result is visual_grounding_probe.json.

DSRL tuning result (Goal OOD 6, seed 173)
Increasing SAC replay updates from 200 to 600 per collected episode yielded
4/10, 4/10, 5/10 and 6/10 autonomous successes at 0, 2, 4 and 8 collection
rollouts (0, 422, 1042 and 1455 collection control steps). Native pi0.5 scored
5/10. The final model gained reset 21 and retained all five native successes;
exact reset-state/model/task hashes agree. The intermediate 5/10 is not the
same set of successful resets. Five of eight collection episodes succeeded.
This is one development seed, not confirmation. No method has reached the
predeclared 8/10 threshold. See dsrl_more_goal6_seed173_result.json.

V5 two-rollout screen completed without an intervention
The updated policy scored 5/10 on exactly the same five reset scenes as native.
Collection consumed 426 controls, including stabilization, and succeeded on
one of two episodes. Astra requested 37 language-subgoal interventions, but its
111 candidate judgments contained 82 uncertain, 14 tie and 15 loss judgments,
with no wins. No assisted action executed. Twelve useful windows from the
successful native episode produced one policy update. Therefore this result
tests filtered native self-imitation, not successful corrective teaching.
The run used 172 calls and 3,380,126 tokens including cached input. Its stop
after two collections and the scheduled evaluation was fixed before that score.
The complete receipt is pilot_v5_goal6_seed173_result.json. The dashboard now
separates intervention requests, computational preferences, selected proposals,
executed assistance, outcome reviews and admitted windows.

Offline diagnosis feedback ablation
At one saved V5 observation (Goal6/reset0, step150), fresh V6 diagnosis calls
used the same images, history and native proposal. Without rejected-candidate
feedback, Astra repeated the language subgoal. With the three previous
comparisons, it chose a +1.99 gripper delta on actions3–4 to prevent premature
opening. This target passed the original controller bounds. No policy candidate
was generated or physically executed, and nothing was added to training. The
two calls support testing the feedback path, not a physical-success claim;
see diagnosis_feedback_probe.json for their exact outputs and token receipts.

Tuned PPO result (Goal OOD 6, seed 173)
Two frozen-actor collection episodes per batch and four optimization epochs
yielded 5/10, 4/10, 5/10 and 4/10 at 0, 2, 4 and 8 collected episodes. The final
checkpoint used 1,686 collection controls and regressed on reset20 relative to
native, with no gained resets. The paired scenes were verified. This is a
development tuning result; ppo_more_goal6_seed173_result.json retains the data.

Prospective fixed-data learner ablation
The V3 collector admitted 22 complete real action windows, but only one contains
a correction; 21 contain native setup or continuation alone. The replay input
loader checks each labeled command against the independent execution ledger,
recomputes admission, verifies original observation hashes and refuses any FRS
steering data. It never fills an unexecuted action tail.

reasoning_replay_v3.json pins the 2.66 MB observation subset by manifest hash.
The ablation starts from the original model, replays the original 20+20 update
order with the corrected seven-channel loss, and then tests 100 and 200 total
optimizer steps using the same real windows. Every measured checkpoint receives
fresh autonomous evaluation; no new collection or Astra calls are made. This
tests the learner under fixed data and does not supply the missing transport
examples. The source 620 collection controls and 4,152,587 teacher tokens remain
attributed to its curves and count only once in the global study totals.

Package this ablation with --reasoning-replay-cache <audited-input-directory>.
Prepare --phase replay-learning --protocol-version v5. The worker checks the
same checkpoint/input/control identity, native-loss parity and reset identity
before training. This phase is not a new independent collection replicate.
