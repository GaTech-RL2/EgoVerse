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
Graphics retries retain HIGH priority and L40S-01; model priority fallback
is unchanged. Only ledger-owned validation jobs are retried.

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
