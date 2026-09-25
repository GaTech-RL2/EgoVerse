# Iterative interventions in frozen pi0.5

This follow-up tests whether Astra's visual and language prior can guide useful
changes to a frozen policy's noise, text embeddings, and camera inputs. It adds
an explicit propose–rollout–feedback–revise loop. The earlier Stage 1 experiment
only inverted numeric action proposals and reused the recovered latent; its
negative results remain separate.

The executable protocol is
[`configs/iterative_interventions_v1.json`](configs/iterative_interventions_v1.json).
It must be frozen in the uploaded payload before evaluating a run. Development
uses standard LIBERO-10 tasks 0 and 1, prescribed state 1. Follow-up evaluation
uses all ten Goal-OOD and all ten Spatial-OOD tasks, with the first captured reset
from seed 19. Earlier seed 7 OOD outcomes are already known. These are new resets
of known task definitions, not evidence of unseen training compositions.

## Search and controls

Each case starts with a policy-generated full internal action chunk, followed by
same-condition flow inversion. The recovered noise must reconstruct the actions
within 0.02 maximum absolute error and the known noise within 0.1. Reference,
inversion, and roundtrip diagnostic solves use RK4, 100 steps, and a cubic time
grid. Actual rollout execution uses the native ten-step Euler sampler in every
arm, in float32 with TF32 disabled. An Euler endpoint is not asserted to equal the
RK4 reference endpoint. Inversion initializes the search; it is not itself a
success optimizer. The reference comes from the policy, which also permits a
direct known-noise control.

The shared first attempt executes the frozen policy with this recovered noise,
reused under fresh observations every five actions. Each unsuccessful case then
receives up to four revised attempts for each of seven intervention combinations:
noise, language, vision, each pair, and all three. Astra gets the original task,
four raw rollout observations, binary success/failure, and previous proposals.
It may inspect its earlier attempts and explicitly revise or restore a candidate.
No object poses, dense simulator rewards, demonstration trajectories, or oracle
phase schedules are supplied. A separate random-noise search uses the same
eight-dimensional space and five-attempt cap. A fresh-noise native policy and a
direct-known-noise reused policy provide single-attempt controls.
Both controls start from the same known latent as the reference; the fresh-noise
control then resamples on subsequent action chunks. The zero-scale embedding
hook must match native Euler output exactly on the allocated GPU.

Every candidate is a complete closed-loop rollout from exactly the same captured
reset. This experiment measures adaptation with simulator reset access. It does
not give a deployed robot uncharged counterfactual lookahead. The underlying
task goal, success predicate, proprioception, and all policy weights remain fixed.

* Noise: absolute coefficients in a fixed, recorded rank-eight orthogonal Gaussian
  basis of the full 10×32 latent; coefficient norm at most one and perturbation RMS
  at most 0.5. The axes have no assumed physical XYZ meaning. Astra can revise
  coefficients and magnitude after observing their consequences.
* Language: add a bounded pooled semantic embedding residual to valid original
  text slots before constructing the prefix cache. Guidance text defines the
  direction. The maximum residual norm is 25% of the original text embedding
  norm; text masks, padding slots, and visual token embeddings stay fixed.
  This tests a specific embedding operator, not every possible language edit.
* Vision: static translucent magenta points/boxes in the original camera pixels.
  Astra selects their locations and gain. There is no tracker; fixed receptacle
  regions are therefore easier to mark consistently than moving objects.
  Astra's feedback images always remain unmodified.

## Iterations and cost

Report success by attempts 1–5, first successful attempt, and intervention
iterations to success (first attempt minus one). Failures at the cap are censored,
not assigned an invented successful iteration. A failed or rejected API proposal
consumes its attempt; it does not silently become a policy rollout or get a free
retry. The ledger distinguishes proposals, executed rollouts, and interventions.

Record provider-reported input, output, total, and reasoning tokens for every
physical API call, including failed calls when usage is available. Reasoning
tokens are a subset of output and must not be added again. Missing usage remains
explicitly unknown. Record cumulative tokens to first success, simulated actions,
flow velocity evaluations (including inversion and diagnostics), and wall time.
There is no assumed dollar price for the NVIDIA endpoint.

Baseline rollout wall time includes reference initialization and recording;
initialization velocity evaluations are counted separately and added once to
each standalone arm cost. Physical worker time also includes artifact uploads.

Development seeks at least one executed intervention per arm even if the first
attempt succeeded, solely to exercise each hook. Rejected proposals still
consume the five-attempt budget. Development also runs a fixed nonzero embedding
probe without executing its actions, because Astra may correctly choose a zero
edit after a successful baseline. Evaluation stops on success.

The common baseline is physically run once per case and attributed to every arm
as its first attempt. Report both physical experiment totals and standalone arm
costs so this sharing does not hide cost. Same attempt limits do not imply equal
GPU or Astra cost; both are measured. A method needs to outperform matched search,
not just its own first attempt, before attributing an improvement to Astra's prior.

## Auditing completed runs

Download each worker's small reports, events, provider ledgers, and reset manifest
into separate folders. Run the recording audit with an explicit phase:

```sh
python -m astra_reversal.intervention_report --phase development \
  --inputs artifacts/interventions-dev-v2/worker_0 \
           artifacts/interventions-dev-v2/worker_1 \
  --output artifacts/interventions-dev-v2-report
```

For evaluation, use `--phase evaluation` and all eight evaluation worker folders.
The output directory must be new. The audit requires every frozen case and checks
reset pairing, event/provider bindings, attempt budgets, and physical costs. Its
JSON, CSV, and Markdown outputs retain success curves, censored failures, and
tokens by attempt budget. Rescue-only iteration and token statistics exclude
cases that already succeeded at the common baseline. Raw-array numerical audits
remain separate from this recording audit.
