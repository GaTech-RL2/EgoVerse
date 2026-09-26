# Astra representation interventions

This is a new experiment following the completed phase-interpolation and pixel
studies. The original FRS evaluation continues with its frozen source and
protocol. It is not modified by this experiment.

The user's September26 proposal, `Untitled (4)`, explicitly treats intervention
choice and admission rules as hypotheses to revise. The first question here is
whether Astra can select useful **text or visual representations** from standard
training demonstrations to rescue the same twenty published OOD compositions.
Assisted improvements must be established before claiming that distillation or
native policy training can preserve them autonomously.

## Comparison

The [machine-readable protocol](configs/representation_steering_v1.json) records
all settings. Each case has one physical shared native rollout, then at most two
revisions per arm: matched native retries; random TLI, VEI and VLI; Astra TEI,
TLI, VEI, VLI, combined TLI+VLI, and raw-pixel donor blending. Failures stay in
the denominator. All methods receive identical captured simulator resets and
the same keyed policy-noise stream at corresponding revision/step indexes.
The random choice stream is separate from the policy-noise stream.

The pilot uses the three previously inspected seed19/reset0 cases. Evaluation
covers all20 known tasks at seed47/reset0. One reset per task is an exploratory
screen, not a precise generalization estimate. There are at most63 pilot and420
evaluation physical rollouts. Native baseline success stops rescue attempts;
development additionally forces one revision per arm to exercise each path,
reporting those extra costs without treating them as gains.

TEI and TLI retain the existing paper-form operators and text banks. VEI blends
the model's projected visual tokens toward a single paired training frame. VLI
does that at post-block visual slots, retaining each camera's spatial token
grid. A single image pair is used rather than an averaged scene. Its bank is
computed under the original target instruction; training task text remains
donor metadata. These visual operators are proposed extensions, not mechanisms
claimed by the text-latent paper. A donor scene can corrupt localization, so
zero-effect parity and nonzero execution checks do not establish usefulness.

Astra receives actual raw observations and labeled donor previews. Every25
actions it chooses a donor/source pair and mixing coefficients, or native
conditioning. The choice is applied to each fresh observation at5-action
replans. The previous completed same-arm rollout supplies raw snapshots and a
binary benchmark outcome. No current reward, object pose, oracle donor mapping
or other arm's history is supplied. Invalid calls clear the active intervention
and consume their slot; there is no hidden retry or stale result application.

## What this changes after the FRS pilot

All18 FRS pilot comparisons were ties, so that experiment performed no policy
updates. This study does not require a promotion before testing a proposed
correction. It compares actual assisted rollouts and retains the evidence even
if a later training experiment rejects it. The base pi05 weights stay frozen;
we will not call these results autonomous learning.

Promising follow-ups are phase-triggered donor choices, more localized visual
edits when global blending destroys geometry, and imitation of executed
successful corrections under the original instruction. Any learned version
must be evaluated with Astra and corrective memory disabled, with earlier
standard tasks retained as a regression panel. These are hypotheses until a
separately recorded experiment tests them.

## Evidence and cost

Each rollout records its captured reset, source/prompt identities, actual Astra
request and accepted response, generated and clipped action chunks, raw policy
observations, solver work, timing, provider usage and an actual20fps video.
Videos show simulated execution time and exclude inference pauses. The source
checkpoint is strictly loaded, native action parity is checked, and parameter
bytes are compared before/after each task. Real weighted vision probes verify
that zero interventions preserve native output and nonzero interventions reach
the action model. The report distinguishes those implementation checks from
task success.

Report baseline, cumulative success after each revision, actions and decisions
to first success, capped failures, total tokens including failed calls, unknown
usage, donor-capture overhead and solver calls. Separate pilot, full screen and
historical settings. Publish representative successes **and failures**, and
label standard-LIBERO footage as in-distribution task examples with unknown
checkpoint training overlap.
