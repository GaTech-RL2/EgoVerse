# Parallel completion of the LIBERO pilots

The 2026-10-12 user request authorizes additional OSMO L40S workers to deliver
LIBERO results the next morning. The delivery target is 08:00 America/Los_Angeles
on 2026-10-13. It does not change the model, prompts, horizon, starts, trial count,
or resource-budget policy.

`parallel_plan.py` partitions the original randomized schedule into an existing
worker prefix and disjoint worker subsets. Every matched task/start/replicate
block stays together, and its method order is preserved. The plan requires
complete coverage without duplicate ownership and pins the full schedule and
the experiment design independently of machine-specific runtime receipts.

`prepare_parallel_osmo.py` packages the exact parent's Git source revision and
adds only these orchestration tools and their scheduling tests. Every Python
file under `astra_reversal/hardware_interface` remains byte-identical to the
parent. Workers independently repeat native commissioning and live/replay smoke
checks. `parallel_worker.py` validates the complete schedule and then projects
the two scheduling inputs used by the unchanged launcher onto its assigned rows.
Actor execution, interface tools, model transport and replay code are unchanged.
The execution receipt identifies both the experiment and orchestration commits.

The original live workers have no cooperative stop hook. They retain the active
matched block and one more complete block. An evaluator-owned empty directory
reserves the first unassigned trial before it starts. The existing exclusive
trial-directory creation then stops that legacy loop before actor initialization;
the original finally blocks analyze and archive all completed work. This produces
an expected `FileExistsError` at the documented handoff boundary. It is a scheduling
handoff, not a failed benchmark trial. No process is killed during a trial, no
source file is changed in the live worker, and no completed artifact is deleted.
The reservation and exact partition are recorded in `execution-handoff.json`.

The original F/B0/B cohort and the F/F-adaptive cohort remain separate. Aggregation
must use execution ownership, the exact frozen starts, and independently replayed
outcomes; it must reject duplicates and retain infrastructure failures explicitly.
Completed GPU jobs alone do not establish a complete or valid comparison.

`parallel_results.py` runs independently on CPU. It reads each assigned owner's
immutable trial archive, verifies record hashes, frozen trial identity, event
chains, native replay and experiment source, and reports missing or invalid
records explicitly. Final completion also requires dependency/controller/asset
and other resolved-runtime hashes to match the parent cohort. Every five minutes
it publishes separate cohort CSV/JSON results, plots and a self-contained HTML
ZIP. It publishes a final delivery snapshot when both cohorts are complete or at
the requested morning deadline; that reporting deadline does not stop actors.

## Shared inference rate-limit recovery

The first eight-worker launch passed native checks but encountered a synchronized
HTTP 429 burst. Four adaptive shards were rejected before their first native
action; the original shards stopped during unscored smoke. The existing parent
jobs also suffered two interrupted pilot blocks. All of these attempts and their
replay/archive receipts are retained.

The replacement launch uses `request_admission.py`: eight distinct one-second
request slots, at most one dispatch per worker per eight-second cycle. An HTTP
429 delays the next already-authorized retry by at least 60 seconds, escalating
on consecutive rejections and respecting a longer Retry-After header. The model's
three-retry limit, exact request body, prompts, native horizon and actions remain
unchanged. Waiting never steps the simulator, and every wait is recorded.

The recovery partition retains the 57 complete original-cohort trials and the
two complete adaptive-cohort trials before the interrupted blocks. It reruns
every method in each interrupted block, irrespective of the earlier method's
task outcome, and distributes all remaining matched blocks without overlap.
The four adaptive requests rejected before any action are retained as provider
outages, consistent with the frozen infrastructure exclusion category. Reporting
must expose the earlier attempts alongside the recovered paired results. No
task failure, low score or successful result triggers a rerun.
