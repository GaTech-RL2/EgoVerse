# Method and reasoning interface

The new [implementation plan](../../REPRESENTATION_PLAN.md) and [machine-readable configuration](../../configs/representation_steering_v1.json) are authoritative. The study is assisted correction with a frozen π0.5 policy. It does not train a policy or use a pairwise judge. Any later learning experiment must have its own autonomous evaluation.

## New representation screen

The pilot uses three previously inspected seed-19/reset-0 cases: Goal task 6 and Spatial tasks 2 and 8. It was interrupted when the provider spending cap was exhausted; [verified completed observations and unfinished costs](development/results/index.html) are reported without a cohort success rate. The full screen specifies all 20 known OOD compositions at seed 47/reset 0 and has not launched. Each case has one physical native baseline and at most two revisions in ten arms: native retries; random TLI, VEI and VLI; Astra TEI, TLI, VEI, VLI, combined TLI+VLI and pixel blending. The maximum is 63 pilot or 420 full-screen rollouts. These are budgets, not results.

All arms share captured simulator dynamic state and model geometry. Native full-dimensional Gaussian noise is fresh at each replan, keyed identically across methods at the same case/revision/step. Revision noise changes together, so matched native retries can explain gains available without an intervention. All execute five actions per replan, use Euler 10 and a 300-action limit. An evaluation arm stops at its first benchmark success; a pilot forces one additional revision for implementation coverage and reports that extra cost separately.

Random controls sample donors from the same catalogs and coefficients uniformly on [0,1], within the same coefficient bounds as Astra. Astra can explicitly defer to native conditioning; random controls do not reproduce its deferral distribution. These controls compare selection strategies under a common policy-noise stream, with that remaining difference stated explicitly.

| Mode | Actual intervention | Native or neutral setting |
|---|---|---|
| TEI | Mix two source instruction embeddings in target instruction slots | Explicit native choice; α=0 selects source A, not the original target |
| TLI | Add `(1−2α)(T_A−T_B)` to instruction slots after blocks 0–16 | α=0.5 is zero residual |
| VEI | Convex blend current projected visual tokens toward one paired training frame | α=0 |
| VLI | Convex blend visual slots toward donor slots after blocks 0–16 | α=0 |
| TLI+VLI | Apply the two operators to disjoint text/visual slots with independent coefficients | TLI α=0.5 and visual α=0 |
| Pixel blend | Blend the raw uint8 camera pair with the same donor before encoding | α=0 |

The nine text banks are frozen means over the prescribed standard demonstrations. The 45 paired visual samples come from fixed phases of the first prescribed standard demonstration of each donor task. No OOD demonstrations are introduced. Visual banks use the original current target instruction; the donor task name is descriptive metadata and donor proprioception is not substituted. VEI/VLI preserve camera order and token-grid position. These visual operators are proposed extensions of the text-latent work.

Astra's configured model is `azure/openai/gpt-6-astra`, with **medium** reasoning effort, an 8,192-token completion cap and a 170-second timeout. One decision occurs every 25 executed actions, at most 12 per rollout; there is no retry. Accepted settings apply to each fresh five-action replan until the next scheduled decision. An invalid or failed call clears the edit and returns to native conditioning until the next decision.

The reasoner sees at most four recent unmodified paired observations with robot proprioception, the original task, its last two decisions, nine donor prompts and two contact sheets covering the 45 paired frames. A revision also receives its own latest completed rollout's four raw paired snapshots and binary outcome. It receives no other arm's history, native action predictions, online success signal, object poses, dense reward or oracle task-to-donor mapping. This is bounded observation-based feedback, not access to the simulator's hidden goal state. The [exact finalized system prompt and settings](prompts/representation.json) bind the [client source](../../representation_agent.py); these are implementation evidence, not proof of a completed call.

## Keep historical interfaces separate

| Recorded study | Reasoning / actions | Correction and feedback |
|---|---|---|
| Phase interpolation, seed 29 | Astra medium / 8192; calls every 25 actions, execute 5 | Fixed recovered noise; instruction TEI/TLI; optional temporary visual annotations expire after 5 actions. Failed calls retain prior valid text. Up to two Astra revisions; oracle gets one. |
| Pixel study, seed 37 | Astra medium / 8192; calls every 25 actions, execute 5 | Fixed recovered noise; true RGB donor blend or neutral occlusion. Apply fresh per replan for 25 actions; failed refresh clears edits. Two revisions. |
| Completed FRS development / interrupted full study | Astra medium / 8192; direction/editor calls every 10 actions, execute 10 | Raw policy cameras. The direction role sees an external view plus a separate calibrated guide; direct and finite-Euler FRS arms share that interface. Critique/editor/judge roles support three fixed adaptation rounds and judgment-gated noise BC. |

The historical clients request no-cache responses and bind request identity, configured model, parsed response and provider usage. The [FRS prompt appendix](../frs_policy_improvement/PROMPTS.md) and [exact four-role manifest](../frs_policy_improvement/prompts.json) remain unchanged. The original phase/image clients and their immutable protocols are linked in the source manifest. A higher effort setting or different feedback format would be a new experiment; it is not silently assumed here.

FRS judges receive paired recorded evidence without the simulator success label; environment stopping can nevertheless reveal rollout length. The new representation screen instead receives the previous attempt's binary outcome and has no learning gate. These information differences prevent pooling results across the studies.

## Videos and outcome semantics

Every gallery item is an original archived MP4, copied without transcoding or replay. The recorder saves the unmodified external camera immediately **before each executed action**, at 20 fps: exactly one frame per action. Stabilization, inference pauses and the final post-action image are absent. Playback is nominal simulator execution time, not elapsed wall time including Astra calls. An image-intervention clip shows the real scene evolving; it is not a visualization of the edited policy input.

Each clip records its task, seed/reset, method/attempt, simulator outcome, action count, encoded frame count, archive/member SHA256 and exact summary/audit binding. Historical clips use complete source audits; interrupted-pilot clips use individually verified completed events from a partial archive or the one complete sealed case, with that distinction retained. Success follows the released predicates; it does not establish stable release or a separate visual verdict. The gallery intentionally includes successes and failures and is not a representative sample for estimating rates. Standard LIBERO footage is an in-distribution task example with unknown exact checkpoint training overlap.
