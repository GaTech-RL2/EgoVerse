---
type: idea
tags: [robot-learning, test-time-learning, policy-adaptation, reasoning, preferences]
created: 2026-09-26
updated: 2026-10-02
status: exploratory-proposal
---

# Reasoning-Guided Test-Time Policy Learning

**This is an exploratory research proposal: these are ideas we think could work, not a validated recipe.** The implementing agent should test the assumptions, diagnose failures from observed evidence, and develop a better solution when the proposed method fails. Intervention mechanisms, sampling, training-data selection, update objectives, and suggested budgets are starting choices that can change for a documented reason.

**Success means better sample efficiency than strong, current state-of-the-art RL baselines.** A working implementation, improvement over the initial policy, or eventual task success is insufficient. If the method does not beat those baselines in sample efficiency, the research objective remains unmet and the agent should diagnose the gap, develop a better approach, and test it.

**Explicit exclusion: never use FRS to guide actions toward an expert suggestion.** The team already tried FRS as an intervention and found it ineffective out of domain. FRS is permitted only as an optional component of probability assessment, subject to validation; reconstruction error alone is not a probability estimate. The permission to revise this proposal does not reopen FRS as an intervention.

**Handoff:** Astra denotes the reasoning teacher that diagnoses failures, proposes corrections, and judges candidates. Before experiments, resolve and record the policy/checkpoint, task and action interface, available observations, named RL baseline implementations, evaluation threshold, and authorized experiment budget from the project context. These deployment choices are intentionally left open. Deliver reproducible code/configuration, intervention and training-data records, sample-efficiency comparisons, and a concise account of failed hypotheses and justified revisions.

## Idea

Use Astra to understand failures, suggest corrections, and judge proposed actions before the robot executes them. Guide action generation toward an acceptable correction, execute one selected action sequence or prefix, and learn from useful behavior in that real rollout. Approved unexecuted candidates can also supply teacher labels, with weaker evidence than observed useful behavior.

The goal is to learn unfamiliar tasks with fewer environment interactions than strong current RL baselines starting from the same policy. Test-time learning here means updating the policy during deployment. Candidate generation and comparison happen computationally; the method must not require executing every candidate, resetting, or replaying the situation for confirmation.

## Assumptions

1. **Astra can work out why an attempt failed.** It can use observations and actions to identify a likely cause and suggest what to change. [REFLECT](https://proceedings.mlr.press/v229/liu23g.html) demonstrates failure explanation and corrective planning from robot-experience summaries. Astra's reliability still needs testing.

2. **Astra's suggestion can improve the next attempt.** We can translate its suggestion into action targets and guide the policy toward them. [Real-Time Chunking](https://arxiv.org/abs/2506.07339) demonstrates target-based guidance for flow policies. Using Astra's corrections as targets is our proposed extension, not a result established by that paper.

3. **Astra can judge proposed actions before execution.** Given the current observations, action meanings, and correction rule, it can identify which candidate is more likely to help and admit uncertainty. [GRAPE](https://arxiv.org/html/2411.19309) supports learning from trajectory preferences, but does not establish this stronger assumption of judging actions before their outcomes are known.

4. **The policy can learn the correction and use it alone.** It has enough information and capacity to repeat the improvement after updating its weights. [OLAF](https://arxiv.org/html/2310.17555) demonstrates learning from actions corrected using verbal feedback, although humans provide that feedback.

## Hypotheses

**H1 — Understanding a failure leads to better next attempts.** On unfamiliar tasks, the policy may keep producing similar failures. A specific correction, such as “lift higher before moving toward the bin,” may help it discover useful behavior sooner than ordinary retries.

**H2 — Comparing proposed actions gives better feedback than scoring each separately.** Deciding which candidate better follows a correction may be easier than assigning each a score. Astra's prediction can still be wrong about the physical outcome. Many RL methods already compare actions against expected performance; the advantage of our feedback remains a hypothesis.

**H3 — Learning from corrections beats strong RL baselines in sample efficiency.** The loop should reach a predefined success level without Astra using fewer environment interactions than strong current RL baselines from the same starting policy. This is the main research hypothesis and acceptance criterion. The papers support parts of the idea, not the complete claim.

Variance is a real motivation: [GAE](https://arxiv.org/abs/1506.02438) reduces policy-gradient variance using values and advantages. [π₀.₆* / RECAP](https://arxiv.org/abs/2511.14759) conditions policy training on relative advantage while retaining rewards and a learned value function; it also uses expert interventions. Our hypothesis is that Astra's knowledge supplies better exploration directions and local improvement judgments with less robot experience. This is not a guarantee that teacher guidance always beats RL, which can also use privileged information.

## Success criterion: sample efficiency against strong RL

Choose a task-relevant autonomous success threshold before comparing methods. The primary measure is the number of new environment interactions required to reach it. Report autonomous success versus interaction budget, including both executed control steps and rollout counts. Assisted success alone does not show that the policy has learned the task. Higher final performance after using more samples does not establish a sample-efficiency win.

Count all data-collection interactions, including failed attempts and Astra-assisted actions. Computational proposals and reusing recorded data do not consume new environment samples; report their compute, reasoning calls, and wall-clock costs separately. Use the same evaluation schedule for all methods and separately disclose its interaction cost. Make any additional demonstrations or privileged information explicit.

Select and document strong, current, task-compatible state-of-the-art RL baselines when implementing the study. Give them fair tuning and use comparable tasks, initial policies, observations, evaluation conditions, and interaction accounting. Use repeated comparisons and uncertainty estimates so a lucky run is not treated as a win. Develop revisions on development tasks/runs and confirm the final claim on fresh evaluation. A tie or inconclusive result does not meet the objective; uncertainty may call for a stronger comparison rather than a speculative method change.

## Proposed procedure

1. Observe the current situation and use any available failure history to identify what needs changing. No extra baseline rollout is required.
2. Ask Astra for a likely cause and a correction rule or action target.
3. Start with target guidance to generate corrected candidates. Astra compares them with a fixed unmodified policy proposal from the same observation and accepts clear predicted improvements. Keep policy weights and the comparison rule fixed during this batch. Constrained MCMC and probability assessment are optional extensions, not required gates. FRS may be investigated only for probability assessment, never to generate corrections.
4. Execute only the selected sequence or its next chunk once. Use fresh observations when replanning; do not physically replay alternatives for comparison.
5. Save the full real rollout. Use the relative-improvement gate below to admit corrections for training, and retain helpful setup and continuation as supporting data. Separately label approved unexecuted candidates as synthetic teacher data.
6. Continue normal operation with the updated policy and track how much intervention it needs. Independent success remains the evaluation goal, without a mandatory extra test rollout after every update.

**Example:** After a grasp, the proposed motion crosses the bin rim too low. Astra checks that prediction before execution and requests more lift. Generate a corrected motion, let Astra check it, and execute it once. If the observed movement clears the rim, it supplies a training example without rerunning the original failing motion.

**Our constraint is Astra's pre-execution check:** $C(o,A;r)=1$ when Astra judges proposed action sequence $A$ acceptable under observations $o$ and rule $r$. Keep the rule and comparison reference fixed within a search batch; refresh them when the situation changes. Passing this check means predicted acceptability, not proven task success. A rule should say when its correction is complete: “clear the rim, then move toward the bin,” rather than indefinitely “lift higher.” Binary acceptance alone does not implement H2; comparisons against a fixed reference are one possible way to construct the check.

### Changing rules and allowing multiple interventions

Astra may tighten, relax, or replace its local rule when evidence or the subgoal changes. The task objective and robot operating limits remain the reference. Freeze the rule only within one candidate batch. If it changes, start a new batch or rejudge existing candidates; do not pool results under different rules as samples from one fixed constrained distribution. Tighter rules may leave no acceptable candidates, so revisit the diagnosis rather than tightening indefinitely.

Allow multiple correction episodes per rollout, with one active correction plan at a time. A plan can combine requirements such as “maintain the grasp while lifting.” End it when its completion check passes, then permit another correction if needed. For the first simple pick-and-place prototype, a soft budget of three episodes is a tunable starting point, not a scientific requirement. Reassess or end an unproductive attempt when the budget is exhausted; do not force unassisted continuation merely to meet it. Also bound search time per decision.

Target guidance is the starting intervention mechanism, with different rules and targets. Change mechanisms when evidence shows this one cannot express or reliably produce the required correction. Monitoring alone is not an intervention. Count correction episodes, modified action chunks, and assisted duration separately, so continuous assistance cannot disappear inside a single event count.

The system should persist an intervention log rather than rely on Astra's conversational memory: event ID, trigger and observation/history references, policy and rule versions, target and completion condition, candidate preferences and uncertainty, executed commands, start/end times, observed outcome, and admitted training windows. Give Astra the active event and a short relevant history; revise disproven rules without erasing their original records.

## Starting intervention: guide action generation toward Astra's target

Keep the policy's normal sampling direction and add a pull toward Astra's target. Change only the action components and time window relevant to the correction. This is **target guidance**. Standard [CFG](https://arxiv.org/abs/2207.12598) instead combines model predictions with and without a supported conditioning signal.

For a flow policy with a linear noise-to-action path, using noise at $t=0$ and actions at $t=1$:

$$
\begin{aligned}
\hat A&=x_t+(1-t)v_\theta(x_t,t,o),\\
E&=\tfrac12\|M(\hat A-A^*)\|^2,\\
v_{\mathrm{guided}}&=v_\theta-\lambda(t)\nabla_{x_t}E.
\end{aligned}
$$

Here $x_t$ is the action sequence being generated, $\hat A$ estimates the finished sequence, $A^*$ is Astra's target, and $o$ contains the ordinary observations and task instruction. $M$ selects which steps and components to correct. The gradient passes through $\hat A$ and changes the generated actions; it does not update model weights. Sampling time $t$ is separate from robot execution time.

Use a bounded, time-dependent guidance strength and the policy's actual sampling convention. Targets must match the action coordinates, normalization, and timing. Convert a gripper-position goal before comparing it with joint commands. RTC supplies a starting implementation for this guidance; whether Astra's targets improve unfamiliar tasks remains to be tested.

### Optional probability assessment and constrained sampling

FRS/inversion may be explored only as part of assessing the policy probability of an expert suggestion. **Reconstruction error measures fidelity, not policy likelihood:** even an extremely unlikely action can reconstruct perfectly. A usable likelihood estimator needs the appropriate change-of-volume calculation and numerical validation; Gaussian density at the inverted noise alone is insufficient. If this assessment is unreliable or too expensive, replace it or omit it. Do not use FRS outputs to steer actions or initialize correction search toward an expert target.

If target guidance fails, diagnose why and try a suitable alternative: a clearer language/subgoal instruction, supported TEI/TLI or CFG, or a bounded waypoint/controller correction when the interface supports it. These are alternative interventions, not probability estimators; FRS is excluded from this list.

For a practical alternative check, draw independent base-policy candidates and measure how often they satisfy the correction or lie within a defined task-relevant tolerance of the expert target. This estimates the probability of realizing that intent under the base policy; a small sample may miss rare behavior. Literal action-density estimation is a separate, more expensive experiment requiring a validated flow likelihood calculation. Neither expert-generated samples nor FRS reconstruction alone provides it.

Inspired by [Finetuning with Sampling](https://arxiv.org/abs/2610.02140), a possible target is the frozen policy conditioned on Astra's acceptance:

$$q(A\mid o)\propto\pi_{\mathrm{ref}}(A\mid o)C(o,A;r).$$

**Why sampling followed by SFT may help:** the ideal target satisfies Astra's correction while changing the current policy's action distribution as little as possible in KL divergence. For a fixed observation and rule, provided the accepted set has nonzero probability:

$$
q=\underset{\rho:\;\Pr_{A\sim\rho}[C(o,A;r)=1]=1}{\arg\min}
\mathrm{KL}\!\left(\rho\,\middle\|\,\pi_{\mathrm{ref}}(\cdot\mid o)\right).
$$

Here $\rho$ ranges over candidate action distributions. Among accepted actions, $q$ preserves the policy's relative preferences: if one acceptable motion was twice as likely as another, it remains twice as likely. Our motivation is to produce corrective training examples that fit the learner's existing behavior and may therefore be easier to learn. The reference is the current policy snapshot for that batch, so this does not permanently anchor learning to the original demonstrations.

MCMC approximates this target; SFT or the policy's native flow-matching objective then fits the sampled examples. **The minimum-change property belongs to the ideal local target, not automatically to the trained model.** Finite sampling and training do not guarantee minimum parameter change or preservation at other observations. If accepted behavior is very rare, even the smallest required distribution change can be large. Better robot sample efficiency remains a hypothesis to test against the RL baselines.

Sampling independent policy actions and keeping accepted ones already targets this distribution. When acceptance is rare but an accepted base-policy sample supplies a valid seed, our proposed noise-space MCMC variant may help: with fixed sampler $A=F_o(z)$, target $q_z(z)\propto\mathcal N(z;0,I)C(o,F_o(z);r)$. Propose $z'=z+\sigma\xi$, $\xi\sim\mathcal N(0,I)$; reject failed checks and otherwise accept with probability $\min(1,\exp[(\|z\|^2-\|z'\|^2)/2])$. This is our adaptation, not a demonstrated robotics result from the paper. FRS is not used to initialize this search.

This is random-walk Metropolis, not Langevin or [FRS](https://arxiv.org/abs/2606.13675). Astra's binary check supplies no useful gradient toward a better action. A Langevin variant would need explicit constraint handling or a differentiable surrogate; Gaussian-prior gradients alone supply no task-improvement direction. Frequencies of expert suggestions describe the expert's proposal distribution, not automatically the base policy's likelihood. Many well-mixed conditional samples can estimate averages under $q$; they do not establish physical success or calibrated pointwise action density.

Freeze observation, policy, sampler, and verifier behavior within the chain. A finite chain may mix poorly. Record chain states including repeats after rejection; accepted moves alone do not represent the stated target. Execute one candidate or prefix. Selecting Astra's preferred candidate is a separate execution choice; high likelihood alone does not establish quality, and inverse-noise density is not action density. Do not add another likelihood weight when fitting samples already drawn from $q$.

## Starting policy update: train useful windows across the rollout

**Retain the relative-improvement gate.** Before execution, compare each correction with the saved unmodified proposal from the same observation. Use clear predicted wins as teacher candidates; defer ties or uncertain judgments. After execution, assess whether the intended local change occurred and whether feedback supports or contradicts the correction. Judge progress by the active subgoal: lifting may temporarily move away from the bin. This supplies observational evidence without executing the alternative and does not establish a causal comparison. Supporting setup/continuation need not each beat a separate baseline, but must be useful to completing the behavior. Unexecuted winning candidates remain synthetic labels; one successful execution does not validate the whole pool.

**Center training on interventions without restricting it to the exact intervention states.** Save the entire real rollout, but select what to imitate:

| Source | Training use |
|---|---|
| Useful setup, such as a successful grasp | Include it. |
| Earlier action that caused the problem | Exclude it or supply a corrected target for that earlier observation. Copying it teaches the same mistake. |
| Useful executed correction | Give it strong emphasis. |
| Useful continuation after intervention | Include it. |
| Failed or ambiguous segment | Keep for diagnosis; do not automatically imitate it. |
| Approved unexecuted candidate | Optional synthetic teacher label, explicitly marked as unverified; acceptance is not physical confirmation. |

For the bin task, learning only an emergency lift near the rim may leave the original low approach unchanged. Relabel a suitable earlier post-grasp state with a lift-first target, then learn the useful transfer and release from actual execution. [OLAF](https://arxiv.org/html/2310.17555) provides a precedent for relabeling pre-intervention actions from verbal corrections. [DAgger](https://proceedings.mlr.press/v15/ross11a.html) motivates collecting labels at states the learner visits; it does not imply blindly imitating every learner action.

Keep each target paired with its own pre-action observations. A hypothetical alternative cannot inherit the future observations produced by the executed alternative. Reconstruct actual action windows from executed commands; only executed steps have outcome evidence. A proposed but unexecuted tail remains synthetic. Use complete labels or the policy's supported partial-label training format.

Sample across intervention events and rollout stages before sampling their candidate variants, so a large candidate pool at one observation does not overwhelm other states. Preserve within-pool sampling weights if claiming to fit $q$. More action candidates do not provide more observed states. Start correction-heavy; expand coverage when normal operation reveals missing transitions. No fixed data mixture is required.

Use the policy's existing supervised objective. For the same linear flow convention, with accepted action sequence $A^+$, fresh Gaussian noise $\epsilon$, and sampled $t\in[0,1]$:

$$
\begin{aligned}
x_t&=(1-t)\epsilon+tA^+,\\
\mathcal L_{\mathrm{correction}}
&=\mathbb E\|v_\theta(x_t,t,o)-(A^+-\epsilon)\|^2.
\end{aligned}
$$

This teaches the policy to generate corrected behavior from its normal inputs. Treat collected targets as fixed during this update. Astra's target need not become a student input, but information essential to choosing the action must be available to the student through its observations, history, or instruction. Otherwise exact imitation may be impossible. Guided actions are teacher data, not ordinary on-policy samples for a policy-gradient update.

**Prioritize learning the new correction.** Use
$\mathcal L=\mathcal L_{\mathrm{correction}}+\beta\mathcal L_{\mathrm{replay}}$
with $\beta$ low or zero. Original demonstrations are not part of the default update; start without them ($\beta=0$). If replay proves useful, first consider a small amount of successful correction data from the current task. Here replay means reusing stored training data, not repeating a physical rollout. Track learning through subsequent normal operation.

## Diagnose failures and improve the method

**Failure includes not beating the RL baselines in sample efficiency, even when our robot succeeds at the task.** Every completed comparison that shows no advantage should trigger an investigation of the gap. This refers to method-level results, not treating each failed rollout as proof that the method is worse.

The implementing agent should identify where samples are being wasted: diagnosis, action representation, intervention, verification, timing, data selection, or policy learning. Check whether the comparison itself is reliable, inspect the losing cases, propose improvements tied to the evidence, and implement and test the most promising revision within the authorized resources. Do not stop merely because the original proposal has been implemented. A replacement need not be one of the alternatives already listed here. Record the baseline result, failure evidence, proposed explanation, change, and effect on sample efficiency. If the budget is exhausted without a win, report the objective as unmet and preserve the best-supported next experiment.

Constrained MCMC and probability assessment, including any FRS-based assessment, remain optional experiments. Compare their added value against target guidance and simple sample-and-filter. Text embedding/latent interventions such as [TEI/TLI](https://arxiv.org/abs/2505.03500), or supported CFG, remain alternatives. FRS guidance is excluded even if another intervention fails.

The main unresolved risks are Astra approving plausible but ineffective actions, search becoming too slow for the observed state to remain current, and the student learning recovery without prevention. Use feedback from the same real execution to revise mistaken rules and synthetic labels. A successful episode does not make every action good; a failed release does not invalidate a useful lift. No physical replay is required to make those local observations, but they do not prove superiority over unexecuted alternatives.

Conditioning on acceptance removes rejected behavior; it does not rank or keep improving already accepted behavior. Once a correction is achieved, move to the next subgoal or a more informative comparison. Keep observed-useful, predicted-acceptable, and failed labels distinct in stored data. Low replay weight permits adaptation but does not guarantee preservation of other skills.

Suggested checks are starting points, not a fixed experiment plan: for example, test H2 using comparisons and separate scores on the same saved candidate proposals. Preserve the real-world constraint against matched physical retries, the explicit FRS exclusion, and the distinction between predicted improvements and observed outcomes while revising the method. Aim for reliable success on new test situations; eventual 100% success is not guaranteed.
