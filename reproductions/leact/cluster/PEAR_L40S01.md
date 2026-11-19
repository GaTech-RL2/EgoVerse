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


### 2026-11-19 — finalize native rewards and bootstrap from learnable training tasks

The rolling sampler is live on the retained 40-H100 allocation
`leact-rp-jhh-pear-primary-02a394b6b227-1`; its metadata confirms task/reset SGS
and eight-update family balancing. The per-family batch gate is absent, while
128 mixed-reward groups of eight remain required. A live check at 04:36 UTC
still found zero completed optimizer updates.

A matched warmstart replay had native successes on coffee-button and
handle-press (two of four checked tasks). The production adapter could omit
that terminal grade: on token-budget or parse termination the agent loop
closed the environment and read Episode.success, but the adapter only canceled
the remote session. The replay instead explicitly requested the native stop
verdict. The fix collects that official verdict before cleanup and records a
separate receipt without adding a sampled action, generated tokens or a chunk.
Missing verdicts raise infrastructure errors rather than becoming fake zero
rewards. The number of prior native successes affected remains unmeasured.

The replacement starts from warmstart496 in a new content-addressed run;
pre-repair rollouts remain diagnostic evidence. Exact configuration composition
is running as `leact-rp-jhh-pear-primary-driver-check-8b8c866e3a6a-1` on CPU.
Only after it passes, and only if the old run still has zero optimizer updates,
the controller stages and checksums a replacement child job inside the same
allocation. Any observed update holds the transition for checkpoint handoff.
Simulator allocations remain retained and are reused with the new run root.
Standby promotion also stages current validated sources, avoiding stale embedded
native reward code. Lost submission acknowledgments retry the same job id.

Sampling initially probes the two confirmed MetaWorld tasks, RoboDojo
`general_pickup` and `align_blocks`, and two admitted LIBERO goal/spatial
training scenes. LIBERO and RoboDojo are candidates, not confirmed student wins.
Confirmed native tasks receive a soft early preference that decays over eight
optimizer updates; newly successful control tasks join automatically. The regular
sampler preserves its 20% hierarchical coverage component and the entire admitted
mixture. Selection does not discard failures from mixed groups. These are
training-only signals, and diagnostic generations never enter gradient batches.

The design follows success-guided sampling of learnable task configurations;
SGS reports PPO evidence, so effectiveness in this VLM/GRPO setting is a hypothesis.
PEAR itself starts with LIBERO and later expands to RoboDojo. Sources:
https://sgs-rl.github.io/ and PEAR's run_stage1.sh / run_stage3.sh.
The required repository-wide unit command was attempted and cannot collect
without the local lightning dependency. The research suite passed 673 tests, with three skipped and eight subtests
passed. JavaScript syntax and diff checks passed. Live activation is tracked
separately from configuration and unit checks.


### 2026-11-19 — prevent hosted telemetry from crossing recipe boundaries

After publishing the sampler explanation, a live page check returned no sampler
policy although shared-storage run.json confirmed task-signal-window-v2. The
hosted SQLite record replaced the locally collected record solely by arm name;
its launch-only/older data hid the current recipe. Run composition now compares
the admitted curriculum hash before merging hosted data and prefers the fresher
heartbeat within the same recipe. Hosted ingestion and PEAR_RUN log folding clear
prior optimizer/reward history when that identity changes. Tests cover an old
run with two updates versus a fresh recipe with zero updates, and subsequent
arrival of the correct hosted record without importing old history. This does
not change any training, sampling, action or reward code.


At 04:46 UTC all exact driver compositions passed (CPU, 525.54 seconds).
The controller submitted job `native-repair-fdc5fe8591f322d5` inside the retained
40-H100 allocation, using new run `rp_jhh_pear_moe_warmstart496_full_a5b016aab118`.
All five keeper children were observed running that replacement; the native
simulator allocations remained held. Model startup and simulator retargeting
were still in progress, with zero optimizer updates. The hosted dashboard was
verified to serve the new assets and the current task-signal sampler metadata.
Recipe identity also suppresses old local/hosted metrics during the interval
before a replacement driver writes its first metadata file.


### 2026-11-19 05:41 UTC — recover retained nodes from orphaned inference workers

The latest live check found the 40-H100 OSMO allocation healthy but all training
children idle after failure. The repaired reward/sampler metadata had activated,
but no rollout or optimizer update completed. Two automatic child-job retries
failed with vLLM reporting only 10.26 GiB free against its 67.3 GiB reservation.
The old native-repair activation receipt was historical, not proof of a live job.

All 32 rollout GPU processes still belonged to the original
`initial-63d8bff380c4` job. Each held approximately 69 GiB. Their workflow,
allocation-control path and attempt identifiers were verified from a whitelist
of process environment fields. Ray actor and vLLM descendants had escaped the
process-group-only shutdown; restarting Ray alone did not release their memory.

Added ownership-scoped process cleanup before a replacement starts and after a
job exits. It matches both allocation and workflow plus an explicit attempt,
preserves the current startup attempt, keeper and calling ancestry, checks PID
start times against reuse, and escalates verified remaining owned processes only.
The Linux detached-child check passed and preserved an unrelated process. The
research suite passed 678 tests with four skipped and eight subtests passed;
the local Linux-only test is skipped on macOS and was exercised on the trainer.
The required root unit command cannot collect without local lightning.

While all five children were verified idle, cleanup removed the old job's
processes inside the retained containers. Rollout GPUs returned to 0–4 MiB used
(approximately 79 GiB free). No allocation was canceled. The replacement bundle
keeps the same admitted curriculum, initialization, frozen routers, 160k segment
budget and 128-by-eight optimizer batch. Old logs and zero-update attempts remain
in job history. Dashboard text now gives the current child-job phase before its
historical recipe-activation receipt.


### 2026-11-19 — restore control rollouts, add L40 capacity, and stage 32-group startup

The retained model job resumed real generation after orphan cleanup: at 06:10 UTC
it had 90 completed episodes, nine successes and four usable groups, with zero
optimizer updates. Those successes were procedural reasoning; RoboDojo pickup
returned a fully graded 0/8 group. Thirty-four harness-finalization receipts
confirmed the repaired native terminal-reward path was executing. Neither the
heartbeat's step=1 nor RUNNING scheduler status establishes an optimizer update.

The MetaWorld/LIBERO worker was still allocated but its child had failed with
ETXTBSY while extracting conmon over a binary mapped by a previous job. Native
bootstrap now gives each child its own bounded-length tool/source/socket root,
while preserving the checksum-pinned runtime and existing image-layer cache.
The replacement child was submitted inside the same eight-H100 allocation;
it completed all 156 task checks and reported eight native lanes ready at
06:22 UTC. The model job was not interrupted by this repair.

The user identified additional L40-03 capacity. Its live pool allowed no
privileged containers and had two guaranteed and four physically free GPUs.
Added two one-GPU RoboDojo replicas at HIGH priority and a two-GPU native YAM
replica at LOW priority using the already qualified SDK images. At the first
check, one RoboDojo and the YAM allocation were RUNNING; the other RoboDojo
request was PENDING. Requests are recorded in additional-l40-03-20261119.json.
A controller hook retargets these exact allocations after a model recipe changes,
including deferred retargeting for requests that have not started, without
submitting duplicate scheduler allocations.

The user asked whether 128 groups should be reduced to obtain an actual update.
Staged a 32 retained-group x eight-rollout startup variant. All admitted families,
SGS, rolling family balancing, the exploration component, frozen routers, LR and
native verifiers remain. Target proportions scale to 24 control and eight
reasoning groups per update; they remain rolling targets, never a per-family gate.
Actual selected-family weights renormalize over present families. A complete
32-group batch still needs 256 trajectories and preserves all eight members of
each mixed-outcome group. Constant groups and missing native grades do not count.

The exact CPU command validation is
leact-rp-jhh-pear-primary-driver-check-e46d9f20cb0c-1, covering strict 128 and
rolling 32 configurations on both 40- and 56-model-H100 layouts. The running
128-group job keeps collecting until the new receipt passes. The handoff checks
for observed optimizer steps immediately before submission: any progress is
preserved for checkpoint handoff; zero-update work can switch inside the retained
allocation to a separate content-addressed run. This is a documented batch-size
variant, not an unchanged paper recipe. No optimizer update or improvement is
claimed from the configuration check.

Validation: 693 research tests passed, four skipped and eight subtests passed.
JavaScript syntax, shell syntax and diff checks passed. The required repository
unit command was attempted and cannot collect without local lightning.


The native control fix produced positive student verdicts: an audit of terminal
finalization receipts found four coffee-button successes and five handle-press
successes. The sampled finalizations were 4/4 coffee-button and 5/9 handle-press;
these are provisional training receipts, not fixed-set success-rate estimates or
completed eight-rollout group counts. At 06:29 UTC the live warmstart policy had
141 completed episodes, 19 total successes and five usable mixed groups, with
zero optimizer updates. Fully judged RoboDojo and YAM groups were still all zero.
The student can therefore produce correct native manipulation actions; this is
not yet evidence of an RL improvement or broad simulator generalization.


At 06:32 UTC, after exact 32-group driver composition passed and the old run was
verified to have zero completed updates, the controller submitted
`startup32-cdce159a8d47849a` inside the same 40-H100 allocation. All five child
processes were observed running this job. Its separate run directory is
`rp_jhh_pear_moe_warmstart496_full_6ba760740bcf`; model/rollout initialization
and the first optimizer update remain distinct from successful job submission.
The dashboard retains the 141-episode/19-success pre-update observation in the
handoff explanation rather than assigning those old rollouts to the new run.


### 2026-11-19 07:33 UTC — hourly chat heartbeat and native retry-root repair

The user explicitly requested hourly progress messages in this conversation.
Installed the local LaunchAgent com.rpunamiya.pear-chat-heartbeat with a
3600-second interval. It uses the supported `codex queue --thread ... --message
... --resume` command against this existing conversation, inheriting its model.
The prompt requires live workflow/child-state checks, actual optimizer updates,
rollout successes, useful groups, checkpoints/evaluations, blockers and ETA;
it requires an update even when unchanged. It preserves allocations and later
user instructions, and never creates another heartbeat. The wrapper locks and
persists its send intent before queueing to avoid repeated sends in one hour.
The first scheduled-command test exited zero, returned a queued-message ID and
no resume warning. launchctl confirms a 3600-second interval and last exit zero.
The runtime is local: the Mac must be awake and the Codex service available.

At 07:25 UTC the startup32 run had 246 completed episodes, 30 successes and five
usable groups, with zero optimizer updates. Fully judged groups still supplied
useful signal only from reasoning; native task groups and partial rollouts are
reported separately. Five model children remained running.

The controller was repeatedly failing on the last YAM slot with "Allocation
keeper is not ready". The scheduler retry had created a new allocation whose
actual keeper root was under the newer run, but its ledger retained the older
allocation_build field. An in-container environment/spec/node read verified the
mismatch. Corrected that one ledger reference; future persistent-allocation
scheduler retries now reset allocation_build to their own frozen bootstrap and
drop stale child-job IDs. The controller then completed core worker job
submission (native_workers_launched=true) without interrupting the model or
canceling any allocations. Added a regression test for that ownership boundary.

Validation: 694 research tests passed, four skipped and eight subtests passed.
The required root unit command was attempted and cannot collect without local
lightning. The heartbeat was operationally verified against the real local
scheduler and existing Codex conversation; no GPU was allocated for notification.

### 2026-11-19 07:46 UTC — first heartbeat recovery, simulator retirement race

The first hourly message arrived in the existing conversation. Live verification
found the startup32 model child running on all five retained model nodes, zero
optimizer updates, and no new checkpoints or evaluation events. At 07:44 UTC,
worker diagnostics counted 309 completed episodes and 36 successes. Group-level
retention is tracked separately from partial-group episode totals; these counts
belong to the current startup32 run and exclude the earlier 19-success run.

One additional RoboDojo child and three core YAM children had failed while their
OSMO allocation keepers stayed running. Their exact worker logs identified the
same transport race: workers can read the same abandoned pending request before
one atomically retires it. The losing reader then saw the retired target and
raised an exception, taking down its simulator process. Treat an already-retired
request with no pending source as an idempotent no-op, while preserving the
error for a genuinely inconsistent duplicate. A regression test reproduces the
stale second reader without nondeterministic thread scheduling.

Staged and checksum-verified four replacement native bundles containing that
transport-only repair. Submitted new child jobs inside the same allocations;
native specs, reset banks, model job, retained rollouts and training recipe are
unchanged. Also reconciled the additional YAM plan to its actual scheduler retry
allocation so it can follow the current model run instead of polling an ended
workflow. The local observer was briefly stopped to serialize ledger updates;
no OSMO allocation was canceled.

The 56-H100 low-priority standby had actually started, but OSMO ended it with
controller exit code 2001. The standby maintenance policy now retries this exact
terminal scheduler failure as well as an unallocated queue expiry. Live clusters
and user cancellations remain untouched, and the controller-failure retry also
requires a pinned standby spec. Submitted standby-r7 at LOW priority on H100-02;
its initial scheduler state is PENDING and it does not autostart a model job.

Validation: 696 research tests passed, four skipped and eight subtests passed.
The required root unit command was attempted; the module form cannot collect
because lightning is missing from the local environment. The live recovery
receipts and timestamped heartbeat snapshots are retained in pear-scaled-v1.

### 2026-11-19 08:38 UTC — control groups arrive; suppress retired-request replay

The next hourly check verified the same startup32 model job running on all five
model nodes. At 08:31 UTC it had 694 completed episodes, 127 episode successes,
77 fully graded groups and 27 usable mixed-outcome groups toward the 32-group
first update. These include nine usable MetaWorld groups and eighteen reasoning
groups. Fully graded native YAM, RoboDojo and LIBERO groups had not succeeded;
episode totals also include unfinished groups and must not be reported as fixed
evaluation success rates. Actual optimizer updates, checkpoints and evals were
still zero. By 08:37 UTC collection reached 704 episodes, 131 successes and 28
usable groups. Earlier-run results remain separate.

Thirteen native child jobs had failed inside live allocations. Exact logs showed
the retired-request error, including one job with the earlier concurrent-reader
fix already installed. That fix handled a missing pending source, but retained
the exception when the pending file reappeared alongside its retirement record.
An identical replay of an already retired UID is now removed without executing
another episode or changing the original retirement evidence. A replay with
different contents is retained and rejected for investigation. Both boundaries
have explicit regression tests; missing native grades still never become policy
failure rewards.

Added a controller hook for this specific failed-child condition. It requires a
fresh heartbeat from the current model owner, every model scheduler task running,
the exact failed child ID, a matching error in its actual logs, and an actual
upgrade of the pinned transport source. It stages and verifies a new immutable
bundle with an identical simulator spec and submits inside that same allocation.
Healthy children and already-upgraded children are left running. Regular ticks
repair at most four children; the hourly recovery pass processes the existing
backlog while the local observer is stopped to serialize ledger writes. Model
weights, recipe, groups, reset banks and simulator allocations are preserved.

The 56-H100 LOW standby retry r9 was RUNNING on all seven scheduler tasks at
08:30 UTC, held idle as a backup while the 40-H100 primary remained active.
Validation: 705 research tests passed, four skipped and eight subtests passed.
The required root unit collection was attempted and remains blocked locally by
the missing lightning dependency.

### 2026-11-19 09:31 UTC — first real PEAR optimizer update completed

The current startup32 job emitted completed optimizer step 1 at 09:26:38 UTC
(01:26 Pacific). Verified the immutable step receipt, its finite reduced actor
metrics, continued step-2 rollout messages, and all five running model children.
This is the primary full-model training run, separate from the discarded engine
qualification update and all earlier pre-update collection attempts.

The first selected batch contained exactly 32 groups / 256 trajectories: twelve
MetaWorld manipulation groups and twenty reasoning groups. Policy-gradient loss
was 0.0652131, gradient norm 4.12213, LR 1e-6 and rollout/current-policy KL
0.00494756. That KL is not divergence from the initial warmstart. The actor pass
took 1815.47 seconds (30.3 minutes), weight publication 23.45 seconds, and initial
rollout collection 7660.95 seconds. Total first-step time was 9500.97 seconds
(158.3 minutes), including the simulator recovery period. Peak actor allocation
was 65.93 GiB per reported worker; no nonfinite metric was present.

At 09:31 UTC worker totals reached 973 completed episodes and 207 successes,
with 47 usable groups cumulatively. Thirty-two had fed update 1; the sampler
reported fifteen finished usable groups for the next 32-group batch, plus
eighteen running groups. Do not show 47/32 as next-batch readiness or interpret
cumulative sampling reward as a held-out improvement. Of the fully graded
groups, MetaWorld supplied 26 usable groups and reasoning supplied 21; other
simulator suites had no usable mixed-outcome groups in this snapshot.

All twenty current native children were running. The repaired request-replay
failure was not recurring in the controller's latest check. One YAM scheduler
allocation had been replaced automatically after an infrastructure failure; its
new embedded initial child was live even though the queue omitted a child ID.
The 56-H100 LOW standby remained pending at retry r12. No allocation was canceled
and the primary training recipe and model job were unchanged.

No primary checkpoint or evaluation had completed. The pinned launch saves every
four optimizer updates. The upstream paper config has test_freq=-1 and disables
in-run validation; do not describe evaluations as automatically underway. The
next checkpoint is due at update 4. Runtime receipts, first-step metrics and the
live dashboard retain the measured timings for revised estimates.

### 2026-11-19 10:36 UTC — second batch ready; resource-monitor retries persist

The same model job remains running on all five model nodes, with all twenty
current native children running. At 10:31 UTC the run had 1402 completed episodes,
294 successes and 63 cumulative usable groups: 32 used by completed update 1 and
31 collected toward update 2. At 10:35 UTC that next batch reached 32 usable
groups, with 1438 completed episodes and 302 successes. Completed optimizer
updates remained one; a full batch is not itself a completed gradient update.
No primary checkpoint or eval had completed. The LOW 56-H100 standby was pending
at retry r15. An earlier scheduler-list HTTP 503 had cleared; current controller
checks were succeeding. No live allocation was canceled or recipe changed.

Episode-level native infrastructure failures continue despite live workers. The
latest transport errors report an invalid resource-enforced verdict. Audited
recent retained native trial JSON: affected MetaWorld trials have
resource_monitor_missing=1 and infra_error=1 while resource_enforced=1. Some
also report live_success=1 / success_before_limits=1. These are rejected and
retried, rather than treated as valid successes or policy failure rewards.
Other recent trials have valid monitor records and usable native verdicts.
The underlying monitor fault is not yet repaired; this is a verified source of
episode attrition. Evidence is saved in hourly-episode-errors.json and
hourly-native-verifier-audit.json. Preserve the grading checks while diagnosing
the monitor; bypassing them would change what counts as a valid outcome.

### 2026-11-19 11:34 UTC — update 2 completed; monitor rejection reasons recovered

Verified primary optimizer update 2 completed at 11:06:30 UTC (03:06 Pacific),
with finite policy-gradient loss 0.0303618, gradient norm 4.62666 and
rollout/current-policy KL 0.00554875. Its learner pass took 1881.48 seconds
(31.4 minutes), collection 4080.45 seconds (68.0 minutes), and total step time
5985.65 seconds (99.8 minutes). The lower loss is a different sampled RL batch,
not evidence of a held-out capability improvement.

At 11:31 UTC the same run had 1623 completed episodes, 322 successes, seventy
cumulative usable groups and two completed updates consuming sixty-four groups.
The replay-buffer log independently reported six finished usable groups toward
the next batch of thirty-two, with twenty-four running groups. All five model
children and twenty simulator children were running. The trainer GPUs were idle
while awaiting the next batch. No primary checkpoint or eval had completed.
The standby r17 queue request expired without starting; automatic retries remain
enabled without stopping any live allocation.

Read the benchmark's unchanged images/monitor/rcb_monitor/rcb_limits.py and
retained verifier/resource_limits.json from actual rejected trials. Two
MetaWorld cases report `16 s of 61 s unobserved` and `7 s of 64 s unobserved`.
A DexJoco case reports that the monitor never found its watched containers
(`no beacon seen`). The resource checker rejects excessive sampling gaps or
missing container scope; the file is not necessarily absent. This narrows the
earlier generic resource_monitor_missing symptom to observed record problems,
without proving why the monitor missed samples or its container beacon.

Saved the source and exact trial detail evidence in
hourly-resource-monitor-source.json, hourly-resource-monitor-detail.json and
diagnostic-sources/rcb_limits.py. These are read-only diagnostic copies. The
resource checks, workload limits, grading, task mixture, and training recipe
were not modified; invalid records continue to be retried and excluded.

### 2026-11-19 12:41 UTC — external idle shutdown and retained-child reconciliation

At 12:31 UTC the current run had 1846 completed episodes, 358 successes and two
completed optimizer updates. Sixteen usable groups were waiting toward the next
32-group batch. The main five model children and nineteen other simulator
children were live; the main native EFE provider was being replaced.

Queried both scheduler records. The previous EFE allocation ended at 12:25:35 UTC
with FAILED_CANCELED and the explicit failure message `Task was canceled by user:
groot-idle-job-shutdown. Auto-cancel due to sustained idleness`. This was an
external automatic cancellation, not this agent releasing or canceling nodes.
The existing controller submitted a replacement, which became RUNNING at
12:32:12 UTC. Its persistent keeper started the embedded initial child. Verified
that its system packages installed and its frozen task-list checks progressed;
do not confuse that bootstrap activity with restored simulator rollouts.

The scheduler retry correctly cleared the old child ID, but the child-recovery
hook never adopted the new embedded ID. Fixed the bookkeeping: adoption requires
the exact initial ID derived from the allocation's frozen spec and the same ID
in its current desired request. This reconciles metadata without interrupting
a healthy child. Also inspect native EFE lane worker.log files when identifying
the already-recognized retired-request failure; its wrapper log alone omits the
underlying lane traceback. Reconciled the live initial children for the EFE
replacement and a YAM scheduler retry. No model or simulator child was restarted
by this reconciliation, and no training parameters or grading checks changed.

Validation: 708 research tests passed, four skipped and eight subtests passed.
The required root unit collection was attempted and is blocked by missing local
lightning. Recovery evidence is in hourly-native-efe-scheduler.json,
hourly-native-efe-recovery.json, hourly-native-efe-bootstrap.json and
hourly-embedded-job-reconciliation.json. Native resource-monitor retries remain
an independent unresolved source of episode attrition.

At 12:49 UTC the replacement's retained Harbor results verified real native
rollouts had resumed: two coffee-button successes and an unsuccessful handle-press
trial, each with resource_enforced=1, resource_monitor_missing=0 and infra_error=0.
The successful episodes executed 208 and 370 native control steps. These are
individual training rollouts, not held-out evaluations or extra successes to add
to the cumulative worker counter. Cold-start recovery evidence is retained in
hourly-native-cold-runtime.json. The controller and all model/simulator allocations
remain running; the external idle cancellation was the only node release.
