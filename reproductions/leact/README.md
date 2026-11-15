# LeAct reproduction

For the November 15 source stack, current EFE/PEAR status, validation and a
replayable dashboard snapshot, see [the review guide](review/README.md) on
the `codex/leact-efe-20261115/04-dashboard` branch. The dated work plan below
records the earlier Leduc and BuilderBench study; its compute limits and
"next" items are historical.

Independent implementation of [LeAct, arXiv:2607.21856v1](https://arxiv.org/html/2607.21856v1).
No author implementation, original prompts, or coldstart artifacts were found in
the paper or the first author's public repositories during this session.

The initial target is the 4,032-information-state Leduc 6r2s experiment. The
implementation follows Algorithm 1 and Appendix A.4: Qwen3-8B, eight backward
candidates, four independent forward baselines, positive-delta top-two selection,
expert-policy targets, and joint forward/backward SFT. ExIt uses forward candidates
and their decoded policies. Evaluation includes single-sample and explicitly
oracle-aware best-of-eight exploitability, with three inference seeds.

## Work plan and status

Updated October 23, 2026. The current sequence is recorded in
[experiment-plan.json](artifacts/experiment-plan.json) and shown on the tracker.

1. **Complete: frontier Leduc control.** Astra explanations with CFR action
   targets, joint warmstart, and two LeAct rounds. The final epoch scores
   **0.007819 ± 0.003090** single-policy exploitability, compared with
   **0.154606 ± 0.068825** at warmstart. All three epochs of each round were
   evaluated with three inference seeds. The retained round-two checkpoint was
   selected by oracle best-of-eight and scores 0.009653 single. These are
   seen-state imitation results from one training run; the earlier BC and ExIt
   results are not matched controls for this frontier warmstart.
2. **Complete: projection-sampled frontier warmstart.** Apply arXiv:2610.02140
   to the same frontier corpus, then repeat the matched warmstart and two LeAct
   rounds. The first two pilots admitted 0/32 rewrites and did not train. The
   current revision proposes only the explanation in non-thinking mode and
   preserves the original CFR action output with a fixed wrapper. Likelihood
   scoring uses the ordinary task prompt and the complete explanation/action
   response; privileged reference prose is absent from that prompt. This is an
   explicit Leduc adaptation. The revised pilot passed 32/32 format/policy/length
   checks and advanced to full sampling: 16 explanations changed and 16 were
   unchanged. A [qualitative audit](artifacts/projection-pilot-semantic-audit.json)
   found five clear cases of semantic drift, including factual mistakes and
   changed recommendations. Fixed solver targets do not imply preserved prose
   meaning. Full sampling admitted all 4,032 traces. The projected warmstart's
   final epoch scored 0.529451 ± 0.081877 single-policy exploitability, worse than
   the original frontier warmstart's 0.154606 ± 0.068825. Round one's final epoch
   scored 0.116175 ± 0.032921. Round two finished at **0.122446 ± 0.038404**,
   versus 0.007819 ± 0.003090 for original frontier + LeAct. Projection did not
   improve single-policy performance in this adaptation. Oracle best-of-eight
   reaches approximately 8.795e-6, near the CFR oracle's own exploitability;
   that performance still requires the oracle to select among candidates.
3. **Complete: self-generated scaling.** Compare 128 states × 1 retained trace,
   4,032 × 1, and 4,032 × 4 from the saved base-model candidate pool. Use matched
   training settings and fixed final-epoch evaluation. The 128-state cell has
   completed all 24 SFT steps and three-seed evaluation: 1.996089 ± 0.061174
   single-policy exploitability. The 4,032 × 1 cell scores 0.539722 ± 0.089349;
   keeping up to four eligible traces per state scores **0.027941 ± 0.014422**,
   a 19.3-fold reduction at the same state coverage. Eligibility leaves 3,904
   states and 11,336 traces in the largest cell (22,672 paired examples).
   All cells use three epochs, so this is a data-and-compute scaling result,
   not a compute-matched comparison. These are self-generated warmstarts with
   CFR action targets, one training run and three inference seeds each.
   **LeAct continuation is a separate arm:** `self-leact-20261024-v1` starts from
   the saved 0.027941 final-epoch checkpoint and runs two LeAct rounds with the
   frontier control's settings. The starting weights and dataset are checksum
   verified. Round one finished at 0.039648 ± 0.012293 and round two at
   **0.022788 ± 0.013215**, an 18.4% reduction in mean versus warmstart. This is
   a small change relative to inference-seed variation and is not an independent
   training replication. Oracle best-of-eight improves from 0.002251 to 8.795e-6.
   Round one admitted 305 traces and round two 251, producing 60 and 48 optimizer
   steps respectively. The warmstart score contains no LeAct rounds.
4. **Next domain: BuilderBench.** Reproduce LeAct's cube-manipulation pipeline
   with GPT-6 Astra (`azure/openai/gpt-6-astra`) as the confirmed expert through
   the existing NVIDIA endpoint. Qwen3-8B remains the planned student. A
   predeclared capability probe completed six tasks with two simulator seeds
   each: 9/12 rollouts succeeded, 5/6 tasks were solved at least once, and no
   outputs were invalid. This small probe establishes feasibility, not full
   benchmark expert quality. A fresh teacher collection now covers all 51
   official tasks with one rollout each and a 200-action limit, using medium
   reasoning effort and an 8,192-token completion budget. Public
   explanations, exact actions, model identity, usage and simulator outcomes
   are retained for every call. The official simulator is pinned to
   `de9130b983236fc4b97ea8121801d09417e410dc`; all 51 tasks are indexed and two
   seeded diagnostic rollouts pass and repeat within 1e-12 numerical tolerance.
   The initial failed hold
   attempt is preserved, and feedback correction uses the unchanged simulator
   and success threshold. These are infrastructure checks, not expert results.
   See the [preflight evidence](artifacts/builderbench/preflight/preflight.json).
   Collection completed all **51/51 rollouts, with 23 fully solved**. The frozen
   collector recovered an API timeout while preserving the failed request and
   its latency. The student pipeline is now submitted as two HIGH-priority H100
   workflows: `leact-builderbench-train-1` and `leact-builderbench-evaluate-1`.
   Scheduler state and actual training progress are reported separately.
   See the [pilot metrics](artifacts/builderbench/expert-probe/gpt6-v2/metrics.json)
   and [collection status](artifacts/builderbench/expert-status.json).

For BuilderBench, first verify the simulator, action schema and expert rollouts,
then freeze the teacher corpus and evaluation split. Compare action-only BC,
joint forward/backward warmstart, CoT ExIt, and LeAct using the same teacher data
and explicit training budgets. Start the LeAct comparison with one round before
adding further iterations. Preserve expert action targets during SFT (EPF), use
the appendix's action-match reward difference for trace selection, and report
single-rollout progress, best-of-64 progress and full-task pass@64 separately.

The first GPT-6 corpus contains 161 action steps from 23 successful tasks. A
matched 8,192-token audit retains 153 steps: 153 action-only BC examples and
306 joint forward/backward examples. The same eight over-context steps are
excluded from both arms, and all 23 tasks retain demonstrations. The other
28 tasks remain outside student training. A second collection on the 23
in-domain tasks (simulator seed 1) supplies separate successful trajectories
for LeAct and ExIt; it does not enter the initial warmstart or change the split.
This first corpus is substantially smaller than the paper's teacher pool.

The training role runs joint warmstart and BC for three epochs at 1e-4, with
effective batch 32 (eight GPUs, microbatch 1, accumulation 4). LeAct and ExIt
start from the fixed final joint checkpoint, use eight candidates/top two,
and train for three epochs at 1e-5. BuilderBench selection follows Appendix
D.2.1's action-match reward gap against one action-only decode with an empty
reasoning block; invalid baselines do not justify positive admission. Positions
are in meters. The appendix does not specify low-level-vector matching, so
that rare case uses exact matching and is an explicit reconstruction detail.

The evaluation role consumes saved checkpoints automatically. Final epochs
receive 64 inference seeds and intermediate epochs three; seed-complete partial
results are labelled separately. All 51 tasks stay in the denominator, including
context overflows and invalid outputs. The simulator has its own environment
with the expert collection's exact dependency versions, and two captured expert
episodes must replay before GPU work proceeds. Each serial decision records
full generation latency and tokens, actions, reasoning, simulator wall time,
and task outcome. No speedup is inferred before these measurements exist.
An additional timing workflow waits until the training node releases its slot,
then measures every checkpoint on the same 153 admitted public observations.
It compares reasoning checkpoints to the reasoning base and BC to an action-only
base, includes invalid-output costs, and reports action-match quality alongside
latency. This separates per-decision generation changes from changed rollout
trajectories while preserving the two-node limit.

The user's standing instruction is to advance through the agreed plan without
another prompt. The controllers submit dependencies, collect results, and retry
quota-canceled allocations with ordinary backoff when cancellation happened
before experiment startup. At most two H100 nodes may be reserved concurrently.

The appendix specifies an 8,192-token BuilderBench context, coldstart training
for three epochs at 1e-4, and iterative SFT at 1e-5 with top-two selection.
Its original teacher pool is GPT-5.2, Claude Opus 4.6 and Gemini 3 Flash; changing
the expert is an extension rather than an identical-data reproduction. Recover
and preserve the published task partition where possible. If it cannot be
recovered, freeze and label a reconstructed partition before training; tasks
with new expert demonstrations cannot silently remain labelled OOD.

The Astra action-expert and three-way Leduc arms remain skipped: full-tree
exploitability 0.51613 failed the user's ≤0.10 gate. BuilderBench needs a fresh
expert competence check using simulator success; the poker threshold does not
transfer. Action-only versus additional-explanation conditioning can be revisited
there after a competent expert and the basic LeAct comparison are established.

Compute remains HIGH priority on gr00t H100s, at most two nodes. BuilderBench
takes precedence over adding larger poker domains. Exact reproduction of the
paper's direct-action BC result remains an open gap, not a completed control.

The distribution-shift hypothesis is that explanations closer to the student's
reasoning distribution are easier to learn from, while LeAct selects useful
student-generated reasoning. Compare the original frontier, projected frontier
and self-generated warmstarts before and after two LeAct rounds. These tests
can support or weaken that explanation; performance alone does not establish
the mechanism. Projection increased student likelihood but worsened both
warmstart and final two-round exploitability versus the original frontier arm. Its pilot also exposed semantic
drift. The largest self warmstart uses more examples and compute. Stronger causal
evidence requires matched state coverage and training budgets, checks that the
explanations preserve information, and independent training replications.

### Rollout speed alongside task quality

Every Astra-supervised student evaluation must retain speed and quality together:
the base Qwen3-8B student, action-only BC, joint warmstart, each LeAct checkpoint,
and the matched ExIt control. The primary deployment comparison uses one live
request and one candidate per decision, identical task/seed cohorts, simulator
revision, history budget and action limits. Best-of-N evaluations have a separate
compute budget and cannot substitute batched throughput for online latency.

Report mean/median/p95 full-response latency, input/output and reported reasoning
tokens, model time per rollout, simulator wall time and total rollout wall time.
Success, cube progress, invalid responses and action counts stay adjacent. The
speedup is reference time divided by student time over the same task/seed pairs;
failures remain in the main comparison, with a separately labelled both-successful
subset. A faster failed rollout does not establish a useful capability speedup.

Astra versus the student measures deployment speedup, including the different
model and serving infrastructure. Base-student versus trained-student comparisons
require the same accelerator/count, precision, engine/version, tensor parallelism,
cache policy, prompts and decoding settings to isolate changes after training.
Record model loading/warmup separately, and include full reasoning generation in
decision latency. Provider-reported cached tokens are retained; requesting no
cache does not establish that the provider used none.

The existing Astra calls already contain monotonic request durations. Their
episode/simulator wall time was not recorded and is left unknown. Subsequent
collector versions measure it explicitly; cached replay episodes are excluded
from total-wall-time speed claims. Running collectors retain their frozen source.

`python -m reproductions.leact.builderbench.speed` rebuilds
[speed-report.json](artifacts/builderbench/speed-report.json) from source evidence.
The tracker publisher refreshes it during collection. Future student evaluations
under `artifacts/builderbench/evaluations/<checkpoint>/` use the same manifest,
per-call and per-rollout evidence contract; the reporter discovers each checkpoint
and compares it to Astra and the base student. Student speedup cells remain empty
until those evaluations exist. The complete protocol is in
[speed-plan.json](artifacts/builderbench/speed-plan.json).

### Historical diagnostics (October 20–21)

Coldstart training is complete. Its 16-state generation gate finished at
2026-10-20 23:29 UTC: all 32 candidates terminated and their conditioned policies
parsed, but all 32 lacked usable reasoning. The quality gate stopped the run
before benchmark evaluation or iterative training. The checkpoint is preserved at
`/mnt/amlfs-07/shared/egoverse/rpunamiya/leact/runs/leact-leduc-qwen3-8b-2/experiment/coldstart/checkpoints`.
The launcher fix for vLLM's shared compiler-cache export race passed runtime
validation on all eight workers in `leact-leduc-qwen3-8b-5`. Checkpoint reads are
warmed on resume. That job passed 23 tests and the eight-GPU NCCL check, then
exited at the reasoning quality gate; it was not quota-canceled. The previous
pool-02 recovery had been reclaimed by the external project quota controller.
The diagnostic completed the original action-only baseline on all 4,032 states,
with eight candidates and three inference seeds. Single-sample exploitability
was **1.107717 ± 0.036289**, and teacher-aware best-of-eight was
**0.303687 ± 0.022365**, versus Table 8's 0.003 ± 0.001 and 0.001 ± 0.0002.
These are final-epoch results from the original recipe, not best-across-epochs.
There were no truncated samples and 16/19/20 parser fallbacks per seed out of
32,256 samples. The original baseline has not numerically reproduced the paper.
The [measurements and comparison](artifacts/original-bc-evaluation/comparison.csv)
are saved locally; the summary and three per-seed JSON files match checksums read
from the running job. The untouched base model's 16-state diagnostic produced
20/32 usable reasoning candidates and 15 positive selections, with 11 truncated
candidates and one policy fallback. A subsequent
[token-length audit](artifacts/base-gate-sft-admission.json) found that all 15
positive selected traces exceed the 2,048-token training limit: their forward
examples require 2,456–4,238 tokens. None would enter the paired SFT dataset.
This is a 16-state diagnostic, not a full-tree score.
The later [baseline-validity audit](artifacts/base-baseline-validity-audit.json)
found that all 32 unconditioned baseline rollouts in that base-model diagnostic
were truncated and parsed using fallback policies. The 15 positive scores must
therefore not be interpreted as improvement over valid baseline reasoning.
The matched prompt experiment reports this validity check separately from the
original selection rule.
The template audit found that the first recipe supervised the empty thinking
block. The BC recipe now sets `enable_thinking: false`, placing the same four
tokens in masked prompt context and training only on the action/policy. This
matches the non-thinking inference prefix and follows the
[LLaMA-Factory 0.9.4 template behavior](https://github.com/hiyouga/LlamaFactory/blob/v0.9.4/data/README.md).
The fresh run found that this correction alone is insufficient to obtain usable
reasoning. All 25 tests,
including two installed-template integration cases, passed in the locked GPU
environment. The corrected recipe uses
`experiment-masked-bc-v1`; the original checkpoint, source snapshot and
evaluation remain separate. Reasoning-bearing LeAct/ExIt targets keep thinking
enabled. Integration tests exercise the installed template's target boundaries.
At 2026-10-21 00:25 UTC (October 20, 5:25 p.m. PDT), the gr00t GPU quota
controller canceled `leact-leduc-qwen3-8b-6` while it was still queued. No node was
allocated, no payload was uploaded, and corrected training did not start. The
startup watcher stopped without resubmitting.
Following the user's explicit HIGH-priority retry request,
[`leact-leduc-qwen3-8b-7`](https://us-west-2-aws.osmo.nvidia.com/workflows/leact-leduc-qwen3-8b-7)
was submitted on one eight-H100 node in `groot-h100-02` at HIGH priority.
It started at 2026-10-21 01:16:37 UTC after 118 seconds queued. The unchanged
payload was uploaded and its remote checksum verified at 01:17 UTC. It passed
all 25 tests and eight-GPU NCCL, then completed 378 training steps with finite
losses (first 2.3127, last 0.0432). The installed template audit confirms that non-thinking
inference and corrected BC training have identical prompt token IDs. The
[training record](artifacts/masked-bc-training-evidence.json) is saved locally.
At 01:33:44 UTC, job7 exited at the reasoning quality gate: **0/32 usable,
18 truncated, zero positive selections**. The other 14 candidates stopped
without a complete thinking block; inspected outputs contain action/policy
fragments or repetition. All 32 conditioned policies parsed. The
[gate result](artifacts/masked-bc-generation-check.json), eight raw shards and
their manifests were checksum-collected locally, and scoring was independently
recomputed. This was a model-behavior failure; the job was not quota-canceled.
The corrected checkpoint remains on Lustre. Full-tree evaluation of this
checkpoint and the iterative LeAct/ExIt comparisons remain pending. No GPU jobs
are active. The next scientific task is to resolve the reasoning coldstart
protocol and measure the corrected BC baseline separately.
The local [run status](artifacts/run-status.json) and
[coldstart evidence](artifacts/coldstart-training-evidence.json) distinguish
completed work from the corrected run and pending iterative comparisons.

The user authorized one to two H100 nodes. Cluster operations use the existing
[gr00t OSMO skill](../../../gr00t/.codex/skills/osmo-agent/SKILL.md)
and the task-specific [cluster instructions](cluster/README.md).
At most two nodes run concurrently. Existing EgoVerse and Astra work is untouched.

## Matched prompt test, October 21

[`leact-prompt-diagnostic-3`](https://us-west-2-aws.osmo.nvidia.com/workflows/leact-prompt-diagnostic-3)
completed on one eight-H100 node at HIGH priority. The original and `concise_v1`
prompts used identical states, checkpoints, seeds, sampling and 4,096-token
generation budgets. This pilot used N=2, M=2 and K=2 on the first 16 oracle states,
all round-zero/player-zero states. It is narrower than the production N=8/M=4
recipe and does not measure full-game performance.

| Model / prompt | Usable candidates | Valid positive traces fitting 2,048 | Fitting 4,096 |
|---|---:|---:|---:|
| Base / original | 20/32 | 0 | 0 |
| Base / concise | 32/32 | 19 | 22 |
| Corrected BC / original | 0/32 | 0 | 0 |
| Corrected BC / concise | 0/32 | 0 | 0 |

Here, a valid positive trace must beat two baseline rollouts that both terminate
and parse, have no selected-policy fallback, and fit both forward and backward
SFT examples. This is an algorithmic admission check, not a semantic correctness
rating. All 32 original base baselines truncated; only one concise baseline did.
The concise base prompt reduced average backward generation from 3,746.84 to
876.09 tokens. Its 19 admitted traces span 11 states and produced 38 paired SFT
examples at 2,048 tokens. The conditional increase to 4,096 was unnecessary, so
no 4,096-token dataset or new model weights were produced.

The [complete result](artifacts/prompt-diagnostic-result.json) was independently
recomputed from all 32 checksum-verified raw shards. Counts, token lengths and
flags matched exactly; numerical differences were below 1.2e-16. All 34 tests
passed in the pinned GPU environment before inference, and job3 verified the
tested Python source bytes were unchanged. The workflow exited successfully.

The [scientific review](artifacts/prompt-diagnostic-review.json) records remaining
limitations. None of the concise base candidates obeyed the literal 200-word
request, despite becoming much shorter. Three inspected admitted examples
contained mistakes about ranks, checking versus calling, or betting order and
pot accounting. The BC checkpoint still produced empty extracted reasoning in
all 32 candidates under both prompts; increasing the SFT context does not fix
that failure. Broader state coverage and a verified reasoning initialization
remain necessary before interpreting this as a reproduction result.

The new prompts, raw reasoning, baselines, conditioned policies, expert policies,
selection scores and SFT length audits are available in the
[local tracker](http://localhost:8126/). New uploads to ITSS are waiting for
renewed NVIDIA/Silverfort SSH authentication; its previously deployed data remains
available. The complete new release can be packaged and uploaded without rerunning
GPU inference.

## Appendix audit and experiment tracker

The [local dashboard](http://localhost:8126/) now opens with separate **Leduc**
and **BuilderBench leaderboards**. Every result includes a plain-language
interpretation and a link to its nearest baseline. LeAct links use the actual
starting checkpoint recorded in training YAML; projection and scaling links
identify their closest experimental controls. Links reveal hidden checkpoints,
highlight the target row, and survive page reloads. Missing baseline measurements
are shown explicitly.

The default Leduc ranking uses final epochs, with selectable single-policy or
oracle best-of-eight scores. Intermediate epochs, diagnostic seeds and earlier
pilots remain inspectable. BuilderBench keeps the six-task pilot separate from
other task/seed cohorts, includes rollout speed next to quality, and leaves
incomplete runs unranked. Detailed narrative, curves and experiment plans remain
under **Training & analysis**. ITSS publication still requires renewed SSH
authentication; the deployment package includes the new leaderboard asset.

The [ITSS dashboard](http://yam-coverage-rpunamiya.nvidia.com:8126/) contains
training loss/LR/gradient norms, workflow history, the executed YAML, and the
[parameter audit](artifacts/appendix-audit.json). Its **Traces & expert** view
contains all 4,032 expert policies and the available base-model and corrected-BC
diagnostics: 16 states and 32 candidates per model, including raw backward text,
extracted reasoning, forward rollouts, unconditioned baselines, parsed policies,
KL, improvement scores, truncation/fallback flags and selection. Base-model SFT
length checks are attached to each candidate. Recorded rollouts are decisions at
information states; complete game trajectories were not generated.

The appendix audit confirms the executed coldstart recipe: three epochs,
LR `1e-4`, effective batch 32, AdamW, cosine with 3% warmup, bf16/FSDP and 2,048
SFT tokens. `4032 / (8 × 4 × 1) × 3 = 378` optimizer steps, with 12 warmup steps.
The paper gives batch ranges rather than a unique 6r2s optimizer-step count.
The iterative recipe is a separate stage: three epochs at `5e-6` is within
Table 11's ranges. Completed LeAct and earlier ExIt evaluations are listed in
the leaderboard; initialization and data differences remain explicit.

Two orchestration mistakes were corrected: BC evaluation no longer depends on a
reasoning gate, and all three saved BC epochs are evaluated before per-metric
selection across seed means. Original final-epoch measurements remain labeled as
such. Section 4 describes a shared frontier-generated coldstart while H.3 lists
4,032 NoCoT examples for 6r2s; the exact mapping to the iterative methods remains
unverified. Reasoning runs now require an explicit checkpoint, dataset SHA256,
source, variant and rationale. NoCoT, CoT and joint variants can be declared;
none is silently treated as an author-verified match.

The tracker is a separate user-owned service under `~/leact-tracker` on ITSS,
with SQLite storage, a user cron supervisor, and port 8126. Imports use SSH;
the web UI is read-only. `tracking/deploy.py` builds and verifies a release.
`tracking/configure_sync.py` configures a key limited to tracker imports and a
Mac LaunchAgent that publishes changed records every 30 seconds. The publisher
needs the Mac online; ITSS continues serving previously stored results when the
publisher is unavailable. Every active GPU workflow also needs the result
collector described in [cluster/README.md](cluster/README.md). No GPU is used
by the dashboard.

```bash
source emimic/bin/activate
python -m reproductions.leact.tracking.deploy --output reproductions/leact/cluster/.build/tracking/deploy
python -m reproductions.leact.tracking.configure_sync --output reproductions/leact/cluster/.build/tracking
```

## Local setup and checks

Run from the EgoVerse repository root:

```bash
uv venv emimic --python 3.11
source emimic/bin/activate
uv pip install -r reproductions/leact/requirements-cpu.txt
python -m pytest reproductions/leact/tests -q
python -m reproductions.leact oracle \
  --output reproductions/leact/artifacts/leduc6r2s/oracle.jsonl \
  --iterations 20000 --tolerance 0.00001 --check-every 500
```

The oracle refuses to overwrite an existing export. Each JSONL has a checksum
manifest containing the exact game parameters, solver history, residual, and
software version. The benchmark teacher reached exploitability 0.00000879513
after 10,500 CFR+ iterations. It is a finite solver approximation, not an exact
equilibrium. The earlier 300-iteration teacher (0.00277394) used for infrastructure
preflight is retained separately under `artifacts/leduc6r2s-preflight-300/`.

For GPU stages use the locked environment and bootstrap in `cluster/`. The
experiment driver downloads a fixed Qwen3-8B revision, runs full-parameter FSDP
training through LLaMA-Factory, then runs eight independent vLLM inference workers.
All project Python tooling runs after sourcing `emimic/bin/activate`.

```bash
python -m reproductions.leact.experiment --root /persistent/leact-run --gpus 8 --phase preflight
python -m reproductions.leact.experiment --root /persistent/leact-bc --gpus 8 --phase full --mode bc
python -m reproductions.leact.experiment --root /persistent/leact-bc-eval --gpus 8 --phase full --mode bc \
  --bc-checkpoint-root /persistent/existing-coldstart/checkpoints
python -m reproductions.leact.experiment --root /persistent/leact-reasoning --gpus 8 --phase full --mode reasoning \
  --reasoning-checkpoint /persistent/explicit-checkpoint --reasoning-provenance /persistent/provenance.json
```

The full default is BC training and evaluation, with three epochs and inference
seeds 0/1/2. An existing BC checkpoint can be evaluated in a fresh report root
without retraining; all three saved epoch weights must remain available.
Evaluation workers load each checkpoint once and reuse it across the three
independently seeded evaluations, saving separate manifests and raw data per seed.
Reasoning mode uses an explicitly supplied shared anchor and defaults to three
rounds, three M-step epochs, and seeds 0/1/2. Only reasoning mode has the 16-state
quality gate before iterative generation. The provenance JSON requires
`checkpoint`, `variant` (`nocot`, `cot`, or `joint`), `origin` (`author_artifact`
or `independent_reconstruction`), `source`, `dataset_sha256`, and `protocol_note`.
Its content hash is bound into the resume protocol. `--rounds` and `--epochs`
control reasoning runs; `--eval-seeds` also applies to BC.
The `oracle`, `generate`, `prepare`, and `evaluate` subcommands also run each stage
independently; see their `--help` output. All exported paths are resolved on the
machine that will train, so do not copy a locally generated `train.yaml` to a GPU node.

## Artifacts and interpretation

Raw generation shards contain every forward baseline, backward candidate,
conditioned continuation, finish reason, and token count. Their manifests bind
the source hash, oracle checksum, model path, sampler settings, seed, and state IDs.
`selection.jsonl` records all scores and admitted traces. `manifest.json` records
truncation, parser fallback, and context rejection counts. Evaluation requires the
complete game tree and refuses missing or duplicate states.

An interrupted generation resumes only under identical provenance. Completed
training stages are reusable; interrupted training requires inspection. Epoch
checkpoints retain model weights, not optimizer state. After all epoch evaluations
finish in an iterative arm, only the selected checkpoint's weights remain; all metrics and checkpoint
configuration files remain, with an explicit weight-retention manifest.
BC retains all epoch weights so each regime can select a different epoch.
The experiment records every evaluated epoch and chooses the next round's
checkpoint by mean oracle-aware BoN exploitability. Inference-seed variation is
reported separately from training replication (one trained model per arm).
`comparison.json` and `comparison.csv` compare measured exploitability with
Table 8. Each metric selects its best checkpoint by mean across all inference
seeds, with checkpoint and round recorded. These comparisons preserve the
reconstruction caveats and do not automatically declare numerical reproduction.
To regenerate them from downloaded results:

```bash
python -m reproductions.leact.compare --results /path/to/results.json --output /path/to/report
```

## Reconstruction choices and open gaps

* Prompts, qualitative-probability bins, exact game betting parameters, and the
  probability parser were reconstructed. Matching the state count does not prove
  every unpublished choice matches the original experiments.
* The parser normalizes rounded unit mass within 0.02. Malformed policies use a
  legal parsed action; if that too is absent, the first legal action is used and
  flagged. KL is capped at 10. Scores average baseline probabilities before KL.
  Strict KL is sensitive to the finite CFR teacher's tiny positive masses:
  rounding this teacher to three decimals yields exploitability 0.000522870,
  but reaches the KL cap at 2,018 of 4,032 states because some probabilities
  become zero. This diagnostic is saved in
  `artifacts/teacher-rounding-diagnostic.json`; no smoothing or pruning is
  silently applied to the experiment.
  A separate [precision diagnostic](artifacts/policy-precision-diagnostic.json)
  shows that six significant digits preserve positive support and the teacher's
  measured exploitability while reducing median answer length from 55 to 34
  tokenizer tokens. This is a possible future serialization ablation; submitted
  training targets retain full precision, and the authors' precision is unknown.
* Truncated candidate generations never enter SFT. Overlength pairs are rejected
  together instead of silently truncating their action targets. These choices and
  rejection counts are explicit deviations from unspecified preprocessing.
* The first experiment implicitly reused the action-only, full-coverage 6r2s BC
  checkpoint for both reasoning arms. This was an unverified initialization
  assumption. The revised driver requires explicit provenance, and the old
  failed reasoning gates are not treated as BC evaluation requirements.
* The paper's Algorithm 1 specifies joint SFT; Appendix C also discusses a
  forward-only variant. This implementation uses Algorithm 1. It does not assume
  the paper's monotonic-improvement claims hold for approximate selection.
* H100s replace the reported H200s. CPU and GPU environments use different NumPy
  versions to satisfy vLLM, while sharing the same exported oracle and OpenSpiel.
* Liar's Dice, three-player Leduc, Flop Hold'em, rank-held-out
  generalization, and the paper's ablations are not implemented yet.

No trained benchmark result is claimed until the corresponding GPU artifacts and
full-tree evaluations exist. `preflight/result.json` is only an infrastructure check.
