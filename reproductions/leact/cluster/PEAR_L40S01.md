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
