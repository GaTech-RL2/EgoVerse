# PEAR graphics-pool migration

The user selected HIGH priority on `groot-l40s-01`. That pool mounts LFS,
whereas the qualified H100 model workers mount AMLFS07. The training mixture,
reward definitions, checkpoint and frozen-router policy remain unchanged.

## Development log

### 2026-11-18: rollout placement and filesystem diagnosis

The previous 40-H100 allocation loaded its inference replicas but started only
152 of 160 native proxies: the fourth rollout node fitted 32 rather than 40.
Advertising 64 Ray CPUs per rollout node fixes the measured CPU reservation
shortage. A prelaunch capacity check includes the native, reasoning and
conservative inference reservations. The following allocation expired in the
queue before startup. Neither attempt performed a primary optimizer update.

A one-GPU probe confirmed L40S-01 availability and its LFS mount. AMLFS07 is
not mounted there. Existing LFS simulator assets were an older release and
failed all 43 required patch checks, so they were not substituted.

`efe_pear_asset_closure.py` resolved the admitted native USD graphs to 122
required files. A zero-GPU source-side job uploaded their 141.8 MB archive
and manifest to private object storage, then fully read both back. A zero-GPU
L40S-01 restore job checked every file and the original release/patch pins
before publishing the private local cache. The five source/runtime payloads
were uploaded separately and SHA256 checked. OSMO rsync's destination is a
directory: specifying a filename nested each upload one level too deeply;
only those verified study-owned files were promoted to the expected paths.

The first real L40S-01 native reset found a runtime-created dependency outside
the graph-only roots: `isaac/Environments/Grid/default_environment.usd`.
The static copy passed its integrity checks; native rollout admission remains
closed until the expanded runtime closure passes a real reset. The user also
authorized a concurrent HIGH-priority `groot-l40-03` attempt, which uses the
original qualified asset tree on AMLFS07.

`efe_pear_object_transport.py` connects the model and graphics filesystems
through a private object prefix. The graphics side retains the existing
native file protocol, atomic episode claims, JSON commands, exact observation
bytes and official binary rewards. Infrastructure errors remain lost
episodes, without fabricated training rewards. Credentials are not embedded
in frozen bundles or receipts.

## Validation and admission

`monitor_efe_pear_l40s01 --watch` requires verified asset/source receipts,
a real cross-pool YAM reset/RGB/action/finalized-verdict receipt matching the
current transport SHA256, and one completed migration smoke for each of
RoboDojo, RoboLab and YAMLab. It then writes `l40s01-storage-ready.json`.
The separate primary observer waits for that gate before resubmission.
Graphics retries retain HIGH priority and L40S-01. Explicit H100 HIGH priority
also overrides capacity fallback profiles to retain quota eligibility.
Only ledger-owned validation jobs are retried.

These migration checks complement the original complete task admission;
they do not establish that the student can solve the tasks. Primary training
has zero optimizer updates until telemetry records an actual update.

IRIS/DeltaCAT is authorized as a fallback if direct placement fails.
Internal design documentation describes compute-pool filesystem mappings,
but current L40S-01 access has not been verified. No IRIS installation,
production-catalog writes or IRIS pipeline launch has been performed.

Runtime receipts and workflow IDs are under `artifacts/embodied-first-exam/pear-scaled-v1/`.
The self-contained tracking page shows input verification, the native
cross-filesystem check and the primary observer's current admission reason.

### 2026-11-18 21:56 UTC: H100 allocation retry

H100-02 allocated five complete model nodes (40 GPUs), with one additional
8-GPU native EFE node on H100-01. The trainer remained at OSMO's group-ready
barrier before its Python bootstrap. The native node read the previous
attempt's completion file and exited after setup. Native startup now waits
for the current model owner's fresh heartbeat and cleared completion marker.
The blocked allocation was replaced with the user's larger validated request:
seven model nodes (56 GPUs) plus one native EFE node (8 GPUs), HIGH priority.
At this timestamp the replacement is queued; 64 GPUs are requested, not active.
The primary watcher retries allocation failures using the validated profiles.

The focused research suite passes 603 tests, with 3 skips and 8 passing
subtests. The required repository-wide unit command was also attempted;
collection is blocked by the local environment's missing `lightning` package.
The dashboard deployment was verified after renewing Entra authentication.

### 2026-11-18 22:04 UTC: fresh fallback checks

Renderer migration checks must clear inherited shard receipts and set
`qualification_only: true`. The validation guard correctly rejected the
initial checks because their worker wrapper inherited the primary-training
flag. Fresh one-task checks are queued with separate validation metadata.
The research suite now passes 604 tests, with 3 skips and 8 passing subtests.

The expanded asset copy's fresh AMLFS05 mount repeatedly failed with Lustre
I/O errors before the script started. Its zero-GPU source job was replaced
with a zero-GPU job using the already qualified AMLFS07 asset replica; the
same release, patch and per-file checks apply. The L40S-01 restore waits for
the committed private archive. A fresh native cross-pool check and migration
observer are queued, so fallback admission can complete independently of
the primary HIGH-priority H100 allocation request.

### 2026-11-18 22:50 UTC: preserve the allocation and stop startup exec polling

The primary workflow `leact-rp-jhh-pear-primary-48775f967d9b-1` has five
RUNNING model tasks (40 H100s) on H100-01 at HIGH priority. The separate
eight-H100 native worker is also reported RUNNING. H100 quota belongs to
Pretrain Extension; the L40S allocation belongs to Beyond BC. The current
model allocation must remain in place during recovery.

The trainer has not entered its Python bootstrap: its last user-visible
startup phase is OSMO's group barrier, while the four rollout tasks passed
that barrier. Inspecting the official runtime code identified an interfering
collector: an exec request before the initial ExecStart is ignored by the
user process, while the control process blocks waiting for its exec socket
for five minutes. The local collector repeatedly selected the trainer.
Collection now excludes the primary group and reads its shared telemetry
through a compatible single-task worker. A regression test covers primary
names, renamed primary labels, and the incompatible LFS mount. The publisher
was reloaded; no model allocation was canceled or resized. Recovery still
requires observing the trainer's bootstrap and actual optimizer metrics.

The research suite passes 608 tests, with 3 skips and 8 passing subtests.
The required repository unit command was attempted again and still fails
collection because the local environment lacks `lightning`.

The telemetry reader also now resolves the current primary run directory
from its accepted launch receipt. The preparation plan retains placeholder
paths, which previously hid the content-addressed curriculum run's files.

The L40S-01 native reset exposed a separate missing input: the task-owned
`asset_manifest.json` is not a USD reference. Asset packaging now includes
all regular files inside the 16 selected task directories, including JSON
manifests and runtime sidecars; unselected task directories remain excluded.
Link rejection and per-file checksums still apply. A fresh zero-GPU source
job and zero-GPU pool-local restore are submitted under new immutable
paths. They do not release or restart any model GPU. The updated research
suite passes 611 tests, with 3 skips and 8 passing subtests; native fallback
admission still requires the subsequent real reset/action/verdict probe.

### 2026-11-18 23:30 UTC: keep the model running and broaden graphics placement

The startup barrier cleared at 22:55:26 UTC after removing the interfering
trainer exec polling. The five-node driver entered its full-mixture command
at 22:58:21 UTC. The model subsequently loaded and the first eight student
episodes returned one positive reward and one parse failure. These are tiny
rollout diagnostics, not an evaluation result or a completed optimizer step.
The first training batch is still filling; a heartbeat's current step of one
does not establish that an update finished.

L40-03 worker cancellations were issued by `groot-idle-job-shutdown` for GPU
quota reclamation. H100 allocation remains 40 model GPUs plus the separate
8-GPU native EFE host. All three L40S-01 renderer migration checks passed,
and 1,027 native assets plus the five source archives were restored and
verified. A fresh native YAM reset/action/verdict check uses those inputs.

The user explicitly authorized any HIGH/LOW L40/L40S placement that can keep
the run going. Retain running lanes; use LOW on the compatible AMLFS07
L40-02/03 pools and HIGH on the allocated L40S-01 pool. Extra LOW L40-02 lanes
reuse the same frozen harness, input hashes and episode queue. Renderer
resource requests match the successful 64-GiB migration smokes. Allocation
retries can alternate only the two explicitly compatible AMLFS07 pools.

`efe_pear_file_relay.py` adds an external relay for the current model's file
RPC; it does not change or restart the model adapters. It transfers exact
native observation bytes, JSON actions, binary rewards and original worker
heartbeat timestamps. Tests cover restarts, infrastructure failures, expired
native workers and ownership changes. Before production routing, its real
native check must also return camera frames, one executed action and the
official finalized verdict. `recover_efe_pear_graphics.py` then reconnects
all 16 graphics lanes (24 GPUs) without reallocating the model or EFE host.
Existing working graphics lanes also keep serving; atomic episode ownership
allows the added pool to operate alongside them without duplicating episodes.
The observer waits for matching source hashes and the complete input/renderer
receipts; it keeps the dense arm paused and does not omit simulator families.

The research suite passes 625 tests, with 3 skips and 8 passing subtests.
The repository unit command was attempted and still fails collection because
the local environment lacks `lightning`. Current run metrics remain the
authority for completed updates; no primary RL improvement is claimed yet.

### 2026-11-19 00:34 UTC: restore HIGH-priority capacity and diagnose failed actions

The user confirmed 32 L40-03 GPUs at HIGH priority. Nine queued core requests
were replaced at HIGH priority and eight more renderer workers were submitted;
running model and simulator processes were retained. The last scheduler census
shows 18 HIGH and 10 LOW L40-03 GPUs RUNNING, with 12 HIGH and four LOW pending.
Accepted submissions are not counted as active GPUs.

The model workflow remains `leact-rp-jhh-pear-primary-48775f967d9b-1`, on 40
H100s. At 00:28 UTC its telemetry records 209 completed episodes, seven positive
rewards and zero completed primary optimizer updates. Positive rewards so far
come from procedural reasoning, with no demonstrated manipulation improvement.
Mixed-reward group filtering and per-family quotas remain unchanged.

The separate eight-H100 EFE host failed. Its Podman logs show repeated container
shutdown timeouts, an API broken pipe and failure to shut the API service down.
Archived trial logs also show failed image startup and canceled trials. An unset
egress-sidecar image appears during cleanup; it is not established as the initial
cause. Security and egress controls remain enabled. Recovery request
`leact-rp-jhh-pear-worker-63f1f5a36b7c-1` uses one rather than two lanes per GPU
and staggers startup by ten seconds. It preserves all six admitted EFE families
and requests eight H100s at HIGH priority. This is a contention hypothesis test,
not a confirmed repair or active allocation at this timestamp.

A native YAM trace audit found 62 episode endings caused solely by decimal
bounds differences below 1e-6. For example, `-0.0475` was rejected against
`-0.04749999940395355`, although both produce identical float32 controller bits.
`efe_pear_float32_json.py` exposes short decimal contract bounds and observed
joint values that round-trip to the original native float32 bits. The strict
action parser still rejects substantive limit violations, wrong dimensions and
unsupported action types. No action is clipped and no physics limit is changed.
The version is opt-in for new workers. A real simulator reset, four nonblank RGB
cameras, one executed boundary action and an official finalized verdict are
required before the observer adds corrected workers alongside existing lanes.
Source hashes bind the qualification receipt to the deployed numeric adapters.

The earlier cross-pool probe itself omitted the parser's mandatory `summary`
field. Its hold and stop commands now pass through the same strict parser used
by the policy. Transport validation and numeric qualification remain separate
from learning results. The independent LFS path remains a fallback while HIGH
L40-03 uses the original admitted AMLFS07 inputs.

The RL engine is PEAR's patched verl stack with vLLM rollouts, Ray coordination
and FSDP full fine-tuning. SGS, score centering, per-family balancing, adapters
and metric collection extend it. The MoE configuration disables LoRA and freezes
expert-selection routers. The current research suite passes 633 tests with
three skips and eight passing subtests. Repository unit collection still fails
because `lightning` is absent. The experiment-journal checkout/helper is absent
from its configured/default locations, so these notes and the self-contained
PEAR dashboard retain the investigation and live qualification state.

The real numeric qualification then completed in
`leact-rp-jhh-pear-worker-5985ad7c941e-1`: four camera images, exactly one
executed native boundary action, a finalized native verdict and matching
numeric-adapter source hashes. This validation deliberately stops after one
step and does not assert task success. The observer submitted eight corrected
two-GPU YAM workers on the compatible LOW-priority AMLFS07 path. Existing
workers remain in service; atomic episode claims prevent duplicate ownership.
A redundant second qualification request was released after the first proof
completed, and its separate submission/cancellation receipts are retained.

### 2026-11-19: diagnose missing learning signal and stage the approved recipe

The 496-step warmstart's completed native evaluation has four successes on
24 seen training scenes and zero on 12 same-simulator new tasks. Its four
successful cases are MetaWorld door-lock, coffee-button, handle-press and
plate-slide-back. Across the 36-case panel, 3,443 of 3,456 requests produced
valid actions (99.6%). These establish usable control behavior on that panel;
they do not establish a warmstart improvement over base, which had four seen
and one new-task success. Native GR00T YAM was outside that panel.

At the subsequent rollout audit, only four of 26 judged GRPO groups had mixed
rewards, all in procedural reasoning. Nineteen native YAM groups, or 152
rollouts, had zero successes. The original exact per-family batch quotas block
the first update even if MetaWorld starts supplying useful groups. SGS was
being applied within reset variants after selecting a task; it cannot change
which single-reset MetaWorld task is chosen.

The user approved two changes, implemented as opt-in `task-signal-window-v2`:

1. Choose tasks using their training-only success histories, and direct more
   dispatch effort toward families yielding informative groups. Family, task
   and reset selection retain exploration components. Task sampling covers
   simulators with equal initial weight, independently of their reset counts.
2. Fill the complete 128-by-eight optimizer batch from available mixed-outcome
   groups. Select groups to reduce family coverage deficits over eight updates.
   Missing-signal families remain in exploration without blocking the learner.
   Normalize loss weights over families present in the batch; log actual group
   counts, target weights and active weights. Save selection history and RNG
   with the curriculum checkpoint so restart preserves the balancing state.

GRPO's constant-reward filter, score centering, learning rate, full-model
frozen-router update and the 160,000-response-token episode budget remain
the recipe. All 1,761 admitted training configurations and protected splits
are retained. Target mixture proportions are rolling targets, not a claim
that an all-failure family contributes gradients. Transfer from MetaWorld to
native YAM remains an experimental question.

PEAR's quick-start allocation table uses **nodes**: eight trainer and 24
rollout nodes at eight H100s each, or 64 plus 192 GPUs. This study's current
model allocation is one trainer node and four rollout nodes, or eight plus
32 GPUs. A sampled rollout node reported zero GPU utilization on all eight
GPUs in six consecutive samples while environment calls were waiting.
Additional model inference GPUs cannot resolve these stalled simulator calls.
Reference: https://github.com/mit-han-lab/pear/blob/main/docs/getting_started/quickstart.md.

The transport queue also retained canceled and expired requests after their
clients had timed out. A reviewed cleanup retired 46 requests into their own
episode directories, preserving their bodies and a reason receipt. No policy
reward was generated. New clients declare their open deadline; workers retire
abandoned requests before starting a simulator. Legacy EFE's 1,800-second and
native YAM's 14,400-second open timeouts are distinguished, with a five-minute
clock/cleanup margin. A zero-GPU maintenance job now performs this cleanup.

The reduced-concurrency EFE host subsequently failed too. Its policy killed
all lanes when one lane exited after five consecutive errors. The updated
worker retries recognized `NativeInfrastructureError` failures with bounded
backoff, retaining error records; unexpected source errors still fail. A new
eight-H100 host includes this change and orphan-request retirement. This
repairs failure propagation and wasted dispatch; underlying native startup
reliability still needs observed rollouts, rather than an assumed recovery.

The four known-success cases are queued for a diagnostic against the current
rollout server. An initial zero-GPU L40 client could not reach the H100 private
network; its four connection timeouts carry no policy grade. The diagnostic
client now runs inside the existing H100 cluster, adding no model allocation.
It uses the current action prompt/parser and native verdicts. Its HTTP path is
a case sanity check, not a claim of identical training-sampler tokenization.

The replacement recipe is staged, not active at this entry. Its CPU check
composes both allocation profiles under both old and new sampling policies.
The primary observer can advance automatically only after this check and a
valid-action native-success replay pass. It prepares the complete new bundle
before releasing the old allocation, and refuses this warmstart handoff if
the old run has completed optimizer updates. Replays and all retries remain
diagnostics, separate from primary learning metrics and benchmark scores.

Validation for this replacement: 647 research tests pass, with three skips and
eight passing subtests. The real CPU launch check passed both sampling policies
under both node profiles and retained matching trainer/config/curriculum source
hashes. Repository-wide unit collection was attempted and remains blocked by
the missing local `lightning` dependency. Native known-case outcomes remain a
separate activation check; CPU validation does not establish policy success.

One diagnostic-specific parser mismatch was caught before interpreting its
plate-slide-back result: the student generated a 138-token reply ending in a
valid four-dimensional `ee_delta` action, but the diagnostic passed the entire
reasoning-plus-JSON reply directly to the strict native parser. Production
already calls `final_action_json` first. The corrected diagnostic reuses that
same decoder and validates native bounds afterward. The rejected wrapper in
the earlier attempt is not evidence of a policy regression. All old records
are retained, and the corrected replay has separate source pins and outputs.


### 2026-11-19 02:10 UTC — update is still awaiting usable groups

Live scheduler queries confirm the 40-H100 primary, the separate eight-H100
native EFE host, and the zero-GPU transport maintenance job are RUNNING. No
primary optimizer-step artifact exists. A fresh rollout-counter snapshot shows
325 completed episodes, seven positive rewards, 39 fully judged groups and
four mixed-outcome groups. There were also 81 infrastructure/runtime failed
attempts; those are distinct from policy failure rewards. The four usable
groups are insufficient for the preserved 128-group batch, even without the
old per-family gate. Native YAM accounts for 30 of the judged groups.

The known-case replay reached 18 valid actions and zero invalid actions in
handle-press, then received an HTTP 400 with no captured response body. The
last successful request already contained 149,086 prompt tokens. The serving
API confirms a 176,384-token context limit, and its tokenizer endpoint passed
a text probe. The exact HTTP rejection cause is not yet established: these
observations motivate bounding each completion to the available context and
retaining the server's full error body rather than treating the incident as a
policy failure. The corrected replay tokenizes its full conversation before
sampling, preserves images/history, and finalizes through the native verifier
when the context is exhausted. Its diagnostic outputs remain separate. The
primary 160,000-response-token recipe and passed driver source hashes were
not changed.

A new replay was launched on the existing rollout node without another model
allocation. Its previous attempt and all results were retained. Two temporary
native diagnostic lanes were requested on the existing simulator host to claim
only these four known training requests, avoiding the general simulator queue;
this adds no GPU allocation and leaves ordinary worker ownership intact.

Validation: 648 research tests pass, three skip, and eight subtests pass.
The regression check covers long-context completion capacity and preservation
of HTTP error details. The required repository unit command was attempted;
`python -m pytest tests/unit -q` cannot collect without local `lightning`.
The journal checkout is absent at its configured/default locations, so this
entry remains in the existing experiment log. No RL improvement is claimed.


### 2026-11-19 — retain allocations; add a LOW-priority backup

The earlier recipe transition canceled the healthy 40-H100 allocation after
its native sanity check passed. That released scarce GPUs, and the replacement
then timed out in the scheduler without starting. This was an orchestration
mistake: a validated recipe change did not justify giving up the allocation.
The user explicitly instructed that failed runs must leave the cluster
allocated and that another job should run inside those same nodes.

Automatic recipe changes now submit versioned jobs inside an existing
allocation. The native-worker cancellation helper preserves allocations by
default; explicit user release is required to cancel them. A persistent keeper
is the main process in each new production task, with training or simulator
work as its child. A failed or completed child leaves the allocation available.
A real CPU process test ran a failed child followed by a successful replacement
under the same keeper PID, then verified explicit release. Peer failures stop
sibling jobs without exiting their allocation keepers. Qualification jobs
continue to finish normally rather than reserving diagnostic GPUs indefinitely.

Jobs are staged and SHA256-verified before switching. Job generations have
separate Ray authentication/startup directories, credential paths, logs and
checkpoint archive identities. Replacement exec is forbidden until every
OSMO task is RUNNING, preserving the startup-barrier fix. Native slots can be
reused across model jobs. Failed startup jobs with zero optimizer updates may
retry twice inside their current allocation; jobs with observed updates hold
for checkpoint recovery instead of silently restarting at the warmstart.

The user also requested a 56- or 48-GPU LOW-priority backup. Submitted 56 model
H100s (one trainer node plus six rollout nodes) at LOW on groot-h100-02 as
`leact-rp-jhh-pear-primary-f0f6d4bebad3-1`. It starts in standby. The prior HIGH
request ended with FAILED_QUEUE_TIMEOUT and zero allocated nodes; a new
persistent 40-model-H100 HIGH request was submitted on groot-h100-01 as
`leact-rp-jhh-pear-primary-9178f4099349-1`. Neither request was canceled to make
these changes. At the latest scheduler check both requests were PENDING.
The controller selects an allocated cluster for the experiment, and a locked
shared writer receipt prevents the primary and backup from writing the same
run concurrently. A later explicit stop/release instruction takes precedence.

The preference is recorded in this study's AGENTS.md and the user-requested
future-reference memory note. No new optimizer update or capability gain is
claimed. The 160k budget, frozen-router policy, 128x8 batch, task curriculum
and rolling family balancing remain unchanged. The standalone journal checkout
is absent, so the development record remains here and on the existing tracker.

Validation for the allocation-lifecycle change: 655 research tests passed, three skipped and eight subtests passed. The required repository-wide unit command was attempted and cannot collect without the local `lightning` dependency.


### 2026-11-19 03:54 UTC — the retained cluster is producing rollouts

Live OSMO checks confirm all five model tasks on 40 H100s are RUNNING in
`leact-rp-jhh-pear-primary-02a394b6b227-1`. The keeper children are active and
the run metadata confirms task-and-reset SGS, rolling family targets over
eight updates, frozen routers, score centering and 128 groups of eight.
Model servers completed CUDA-graph startup, followed by actual rollout
completion records. At 03:54:07 UTC the counters were 95 completed episodes,
eight successes, 24 parse endings and eight separately recorded infrastructure
failed attempts. There were zero completed optimizer updates. These are
initial warmstart-policy rollouts, not evidence of an RL improvement.

The 56-H100 LOW standby request expired without receiving nodes. A fresh
56-H100 LOW request, `leact-rp-jhh-pear-primary-977d7595149a-1`, was submitted
on h100-02 without changing the running primary. The controller now refreshes
and retries never-started standby queue timeouts even while the primary runs;
it preserves running/initializing requests and honors user cancellations.
Retry submissions count all seven nodes instead of a single simulator node.

The tracker reused one arm-label event-cache file across the strict and rolling
curriculum run directories. Inspection found 142 cached events spanning the
previous evening and the new run, even though the live rollout counters came
from the new run. Cache files now include the admitted run-directory hash, so
historical groups cannot be assigned to the new recipe. Existing historical
cache files remain intact. This changes tracking only, not model jobs or rewards.
