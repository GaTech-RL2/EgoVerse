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
