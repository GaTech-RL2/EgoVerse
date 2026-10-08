# LIBERO universal hardware interface experiment

The supplied [reviewed protocol](PROTOCOL.md) is retained byte-for-byte. This
study compares direct Astra control through a typed interface (F), audited
codebase discovery without an observer (B0), and the same discovery workflow with
a separate visual observer (B). F minus B0 is primary; F minus B is secondary.
There is no pretrained robot policy or policy-weight training in this study.

On 2026-10-08 the user removed experiment resource budgets. This instruction
supersedes the budget clauses in the original protocol, which remains unchanged
as a historical source. The amended manifest uses `null` for elapsed-time,
workflow-token, tool-call, observer-call, output-token and local context caps.
Usage is still measured. All arms retain the 1,000-step episode horizon and the
same command validation. Comparisons now measure completion at a common robot
horizon with unrestricted experiment compute; compute usage is an outcome.
The new cohort restarts with fresh sessions and is never pooled with capped runs.

Implementation lives in `astra_reversal/hardware_interface`. Both interfaces
use the same controller validator, sensor allowlist, per-step success check and
execution proxy. Simulation pauses during model inference. A delayed reply
cannot become stale solely because wall time passed; actions must still refer
to the current simulation step and obey the overall wall deadline.

## Current readiness

OSMO L40S commissioning `libero-hardware-interface-20261008-check-8` completed
on 2026-10-08 using source `e3d6421f8b3767575a65884a840d83e6b1664ac6`.
All nine readiness gates passed: real rendering, matched reset/action/sensor
traces, source and scratch isolation, per-step success and budget guards, and
live Astra trials for F, B0 and B with independent replay. Its Linux/Python
3.8.13 contract suite passed all 26 tests. The revised local suite has 40 passing
tests, including all three arms reaching the full horizon beyond the old token,
wall, tool and observer limits.

The three unscored smoke trials exhausted their token budgets: F executed
40 native steps, B0 35, and B 39 with one observer call. Task success was 0/3.
These receipts establish functioning transport and execution; task performance
and any interface advantage remain unestablished.

Receipts and the exact wheelhouse are preserved under
`s3://rldb/experiments/libero-hardware-interface-20261008/libero-hardware-interface-20261008-check-8/`.
The checked-in [live smoke evidence](evidence/commissioning-check-8/model-smoke.json)
records every gate and replay audit. The smoke GPU has been released.
The historical capped pilot `libero-hardware-interface-20261008-pilot-1`, source
`0121630d4d6c8d23c2064ed0bc713eb6d96500fc`, stopped after 61 trials: 60 budget
terminations and one NVIDIA HTTP 503. There were zero successes. Its artifacts
are archived under its own R2 prefix and its GPU has been released. The new
uncapped cohort requires fresh commissioning of the amended source.
The amended workflow `libero-hardware-interface-20261008-uncapped-1`, source
`2d872a5cf2fd7cefe52210793c4b29c71bff4342`, passed 38 Linux tests and every native,
live transport and three-arm replay check. Smoke B used 120,991 tokens without
a budget cutoff. The pilot then stopped before its first trial because CSV
serialized unlimited budgets as empty cells and the schedule loader did not
restore them to null. The loader now restores only those nullable fields, and
regression tests cover full capped and uncapped schedule round trips. The exact
archived 450-row schedule also passes the corrected comparison. These smoke
checks have zero successes and establish execution only, not task performance.
Confirmation remains gated on the complete pilot and a new power-based freeze.
Validate this isolated study with
`python -m pytest --confcutdir=tests/unit/hardware_interface tests/unit/hardware_interface -q`.
The repository-wide test setup imports optional Aria tooling that this study
does not use. The user authorized skipping that unrelated suite on 2026-10-08;
`projectaria_tools` is not required to run or validate this experiment.

The user identified the credential as an NVIDIA Inference Hub key. It works at
`https://inference-api.nvidia.com/v1` with the returned model identifier
`openai/openai/gpt-6-astra`. Local live checks passed generation, function calls,
front/wrist image delivery, tool-result continuation, and the observer JSON
schema. The original OpenAI HTTP 401 was an
endpoint mismatch. The key stays in a private file outside payloads and artifacts;
the transport rejects credential-bearing redirects and does not pass the file
setting into replay children or scratch processes.

In the historical capped protocol, NVIDIA's pre-generation token counter was
approximate: one tool request counted
55 input tokens but generation reported 133. The frozen gateway reservation is
`2 * estimated_input + 4096`, plus the output cap, checked against remaining
workflow and context budgets. Actual provider usage is charged, including
reasoning, and an actual count beyond the reservation stops collection before
any returned action is accepted. This margin is conservative and empirically
checked, not a provider-guaranteed exact count; any breach invalidates readiness.
It ended trials before the nominal budget was fully consumed. Estimates,
reservations and actual counts are retained separately in those old receipts.
The amended `usage_only` protocol makes no counter or reservation requests,
omits `max_output_tokens`, and charges actual actor and observer usage without
stopping on it. All arms use the same rule.

Provider context/output ceilings remain technical service limits. The amended
request uses `truncation=auto` at the provider's actual context boundary, with
the task repeated in each request's instructions so it survives old-history
truncation. Full request/response history remains in the private event log.
The [Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
documents this behavior; NVIDIA transport is checked live before collection.
A truncated response at the provider's own output limit is an infrastructure
error, not an experiment-budget termination. Individual requests retain a
180-second transport timeout, and scratch processes retain isolation limits.

The observer's frozen system prompt is unchanged. Its fixed evidence requests
ask for at most two short facts, one uncertainty and one occlusion; structured
output uses low verbosity. The historical 256-token cap has been removed along
with the actor's 2,048-token cap. All transport diagnostics are unscored.

Earlier check-6 native receipts and check-7's partial live trials are retained.
Check-7 stopped before B0 could act because the provider rejected `uniqueItems`.
That unsupported API keyword is removed; duplicate-key rejection remains in the
shared execution proxy. Live checks now validate the actual tool schemas for
all three arms before simulator trials. Output-cap exhaustion is retained as a
trial budget failure, rather than misclassified as a provider outage.
The amended template clears historical runtime/readiness hashes. A fresh job
checks its code, dependencies, assets and source view, validates live transport
for every arm, and independently replays three 20-step commissioning trials.
Only the commissioning check uses that short horizon; pilot trials use 1,000.

## Frozen design

- Official LIBERO commit `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`,
  `libero_10`, task order 0, all ten tasks.
- Pilot official states 0–4; proposed confirmation states 5–14; seed 137;
  one fresh actor session per state/arm; randomized paired order seed 20261008.
  This yields 150 unscored pilot trials and 300 proposed confirmation trials.
- Confirmation remains blocked until pilot discordance supports a documented
  power calculation and a new frozen confirmation manifest.
- A separate `libero_spatial` task 0/state 0 is reserved for commissioning.
- 1,000 actor control steps, at most ten repeated steps per action; no
  experiment cap on tokens, calls, context size, output length or elapsed time.
  Reasoning is recorded as part of provider output. Success, the agent's finish
  command or the episode horizon ends a normally functioning trial.
- Cameras are 128×128, upright RGB, at a 20 Hz controller rate. Ten identical
  settling actions occur before the actor budget starts.

The official README prescribes Python 3.8.13. The OSMO image is pinned by digest
to that version. Its unmodified requirements are installed with additional pins
for otherwise unspecified dependencies, including robosuite 1.4.1 and MuJoCo
2.3.7. The explicit Torch variant is 1.11.0 CPU: it only loads official reset
assets, unlike the upstream CUDA policy-training recipe. Exact installed wheels,
wheel hashes, Python/build-tool versions, editable-source SHA, OS packages,
benchmark-asset hashes and controller configuration are archived.
Debian packages use the signed 2024-09-01 snapshot after the live bullseye
security mirror returned missing package URLs during commissioning. Only the
snapshot's expired date check is disabled; package signature checks remain on.

## Isolation and interpretation

Actors can read only a published positive source allowlist. Task definitions,
goal/success/reward code, raw simulator state, demonstrations, weights and other
trials are unavailable. The source view includes controller/robot code, a camera
sensor projection, and two minimal LIBERO wrapper/sensor projections, each with original and projected
hashes. F additionally receives generated channel documentation and typed tools.

Scratch runs in a separate Linux process with a minimal read-only runtime and
source tree, its own writable scratch directory, chroot, UID/GID 65534, no new
privileges, resource limits and seccomp denial of network and process-escape
operations. It receives JSON only. Nested robot calls pass through the same
validator and tool budget. An unavailable OS isolation mechanism blocks execution.

The observer receives only current camera frames/timestamps and one of three
fixed evidence requests. It sees no task instruction, actor transcript, proposed
control, joint state or evaluator label. Output schema and a conservative advice
filter are enforced. A language model is not deterministic; these checks cannot
prove every possible description is accurate or non-advisory. Live validation is
required and observer failures count in B's workflow budget.

Attempted controller-bound violations and applied guard bypasses are recorded
separately from schema errors. Native collision and joint dynamics are unchanged;
no additional workspace limits are invented when the native controller exposes
none. This supports simulation measurements, not physical safety claims.

## Commands

From the repository root, activate `emimic` before Python commands:

```bash
source /path/to/emimic/bin/activate
python -m pytest --confcutdir=tests/unit/hardware_interface tests/unit/hardware_interface -q
python experiments/libero_hardware_interface/tools/launcher.py probe \
  --manifest experiments/libero_hardware_interface/preregistration.yaml \
  --libero-root /path/to/pinned/libero --out /path/to/new/prepared
```

The short protocol commands are supported as well: `--libero-root` defaults to
`src/libero` beside the manifest, `--prepared`/`--catalog` to its prepared
receipts, and the run output to the manifest directory. Explicit paths below
are useful when the source, prepared runtime, and run outputs live separately.

The OSMO bootstrap runs the pinned Linux installation and probe. It writes a
resolved manifest and immutable receipts; model smoke gates remain false until
actual actor/observer trials complete. Use a separately frozen resolved manifest:

`tools/prepare_osmo.py DESTINATION --stage smoke` packages a committed checkout
for an unscored OSMO L40S run. Transfer the private key separately as
`/osmo/run/workspace/inference-api-key`, mode 0600. It is read through
`HARDWARE_API_KEY_FILE`; never include its contents in workflow YAML or source
payloads. The smoke stage repeats native commissioning, tests live image/tool/
observer transport, runs all three actor arms, and independently replays them.
Only then does it write a new `ready-preregistration.yaml`. Pilot collection is
a separate launch after inspection of these receipts.

`tools/prepare_pilot_osmo.py DESTINATION --commissioning RETRIEVED_SMOKE_ARTIFACTS`
packages the fixed 150-trial pilot on one L40S. It requires completed three-arm
smoke/replay receipts and the same adapter and model settings. The new job
repeats native commissioning and must reproduce every resolved runtime hash
before collecting any pilot data. It retains partial failures, writes the pilot
analysis and power worksheet when complete, archives its own artifacts, and
releases the GPU on exit. It never launches confirmation automatically.

For the amended protocol, `tools/prepare_osmo.py DESTINATION --stage pilot`
packages fresh native, transport, and three-arm replay checks followed by the
150-trial cohort in one owned workflow. The three commissioning episodes have
a 20-step horizon; every pilot episode has the full 1,000-step horizon. No
experiment resource cap applies to either stage. The workflow omits a custom
execution timeout and inherits the pool's infrastructure timeout. Confirmation
is never launched automatically.

```bash
python experiments/libero_hardware_interface/tools/launcher.py smoke \
  --manifest /path/to/resolved/preregistration.yaml --prepared /path/to/prepared \
  --libero-root /path/to/pinned/libero --out /path/to/new/smoke
python experiments/libero_hardware_interface/tools/launcher.py validate \
  --manifest /path/to/frozen/preregistration.yaml
python experiments/libero_hardware_interface/tools/launcher.py schedule \
  --manifest /path/to/frozen/preregistration.yaml --catalog /path/to/prepared/catalog.json \
  --out /path/to/new/trials.csv
python experiments/libero_hardware_interface/tools/launcher.py run \
  --manifest /path/to/frozen/preregistration.yaml --prepared /path/to/prepared \
  --libero-root /path/to/pinned/libero --schedule /path/to/trials.csv \
  --split pilot --out /path/to/new/pilot
python experiments/libero_hardware_interface/tools/launcher.py audit \
  --runs /path/to/pilot/runs --out /path/to/new/audit.json
python experiments/libero_hardware_interface/tools/analyze.py \
  --manifest /path/to/frozen/preregistration.yaml --runs /path/to/pilot/runs \
  --split pilot --out /path/to/new/results
python experiments/libero_hardware_interface/tools/launcher.py power \
  --manifest /path/to/frozen/preregistration.yaml --runs /path/to/pilot/runs \
  --out /path/to/new/power-worksheet.json
```

Every actor trial is independently replayed in another process from its official
initial state and applied-action log. Replay verifies first success and terminal
state. Hash-chained events preserve request/response bytes, per-step actions,
model usage and source identity. Raw model transcripts stay evaluator-owned.
Analysis retains all started trials, reports incomplete pairs and unknown cost,
and clusters repeated sessions by official initial state. Infrastructure reruns
must be separately versioned matched blocks; existing trial directories cannot
be overwritten or silently excluded.

The launcher also performs an evaluator-only calibration for every official
initial state before its matched actor block. Initial robot sensors and pixels
must match that calibration in every arm. Initial and terminal camera frames
are retained privately. The live dependency set and benchmark assets are
verified against the frozen receipts before collection.

Analysis exports JSON, per-task CSV, paired-comparison CSV, a success chart and
empirical step/time/token distributions. Provider-unknown usage breakdowns and
cost stay null. Model attempts, failed transports, provider latency, simulator
latency, tool timings and failure codes are retained. Confirmatory claims also
require a passing independent replay audit for every started trial.

The power command requires a complete, audited pilot. Its paired-binary normal
approximation uses pilot discordance, a conservative upper-discordance scenario,
the preregistered target effect, and equal allocation across tasks. This is a
planning worksheet, not exact power for the bootstrap test or permission to
start confirmation. A new manifest must freeze the selected sample size and
verify enough disjoint official states exist. Infrastructure exclusions, if any,
need a separate documented valid-run sensitivity; they are never automatic.
