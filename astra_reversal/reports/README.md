# Review snapshots — 2026-09-24

These compact reports and one original development video make the completed
measurements reviewable without downloading worker archives. They contain no
lossless tensor arrays, raw provider response bodies, credentials, or signed
download URLs. No new inference calls or numerical solves were performed to
create the review copies.

| File | Content | Relationship to worker artifacts |
|---|---|---|
| [ood_baseline.json](ood_baseline.json) | Canonical 86/200 OOD Euler baseline, original 82/200 run with eight errors, complete 25-episode repair audit, protocol and workflow provenance | Unchanged copy of the baseline aggregator's report |
| [ood_baseline_tasks.csv](ood_baseline_tasks.csv) | All 20 task names, denominators, successes, errors, actions, and costs | Unchanged copy of the aggregator's task table |
| [runtime_numerics.json](runtime_numerics.json) | All 14 conditions at each of RK4/100, /200, and /500; full error metrics, costs, native parity, source/checkpoint identities, and artifact hashes | Compact review summary of the 1.45 MB numerical gate |
| [astra_proposal_preflight.json](astra_proposal_preflight.json) | Passed genuine Stage 1 proposal round trip, actual model, request/response/provider hashes, controller and full internal errors | Unchanged copy of the GPU preflight report |
| [astra_development_smoke.json](astra_development_smoke.json) | Exact completed development episode summary: unsuccessful task, 26/28 roundtrips passing, zero fallback | Unchanged worker summary |
| [astra_development_review.json](astra_development_review.json) | All 28 recorded roundtrip metrics and independently checked provider/observation bindings; two numerical failures retained | Unchanged read-only review report |
| [astra_development_smoke.mp4](astra_development_smoke.mp4) | Original 211 KB video of the unsuccessful 520-action development episode | Unchanged worker video |
| [astra_development_replay_plan.json](astra_development_replay_plan.json) | Fixed development cases 310/330/485 and resolutions 100/200/500, with exact original array/provider bindings | Input plan recorded before the separate GPU diagnostic; results pending |
| [payload_source_parity.json](payload_source_parity.json) | Execution/core commit identities versus frozen payload; exact file hashes and three harmless formatting/import differences | Code-identity review, not an implementation-correctness proof |
| [snapshot_sources.json](snapshot_sources.json) | Source paths, source and snapshot SHA-256 values, sizes, and copy/compaction descriptions | Snapshot provenance manifest |

The standard LIBERO-10 455/500 baseline is already tracked in
[checkpoints/libero_l40s_full_baseline.json](../checkpoints/libero_l40s_full_baseline.json).
The interpretation and protocol limits are in [RESULTS.md](../RESULTS.md) and
[NUMERICAL_RESULTS.md](../NUMERICAL_RESULTS.md). Paired RK4 OOD controls, genuine
Stage 1 OOD evaluation remain in progress in this snapshot. The live development
smoke completed with numerical issues: two same-condition roundtrips exceeded
the unchanged 0.02 internal-space limit, and the task separately failed.

The numerical summary preserves all 42 condition/candidate metric rows without
rounding. Common condition provenance appears once in `full_condition_order`,
keyed by `condition_id`. Per-solve grids are omitted; the original report and
their fixed formula are identified. The source's 154-entry array inventory is
represented by its count, byte total, and a deterministic digest. Per-candidate
endpoint archive and tensor hashes remain present. Each omission is described
in `compaction`. Its `review-summary-1.0` schema is deliberately different from
the runtime gate schema. Workers must use the **complete original runtime gate**.

Absolute filesystem paths and relative raw-record filenames inside copied JSON
are original-run provenance. The corresponding arrays, episode JSON, and provider
records are not included here. The numerical preflight's `astra_calls_run: false`
means it replayed a saved genuine response without another API call; provider
acceptance and original response hashes are retained. Its known-noise metric is
for a separate policy-generated sample. Astra's own actions have no known
generating policy noise.

Optional local raw artifacts, available only after downloading the run records:

- [Original baseline aggregation](../artifacts/ood_baseline_report/report.md),
  including its canonical and original/repaired episode records.
- [Complete runtime gate](../artifacts/osmo_runtime_l40s/runtime_diagnostics.json).
- [Original proposal preflight](../artifacts/osmo_ood_steering_group0/astra_proposal_diagnostics.json)
  and [original vision-smoke files](../artifacts/astra_vision_smoke).

Public object URLs containing a `generation` query parameter identify pinned
asset versions; they are not credential-bearing signed URLs.
