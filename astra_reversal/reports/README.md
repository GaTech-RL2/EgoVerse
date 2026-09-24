# Review snapshots — 2026-09-24

These compact reports and original development/OOD videos make the completed
measurements reviewable without downloading worker archives. They contain no
lossless tensor arrays, raw provider response bodies, credentials, or signed
download URLs. No new inference calls or numerical solves were performed to
create the review copies.

| File | Content | Relationship to worker artifacts |
|---|---|---|
| [ood_baseline.json](ood_baseline.json) | Canonical 86/200 OOD Euler baseline, original 82/200 run with eight errors, complete 25-episode repair audit, protocol and workflow provenance | Unchanged copy of the baseline aggregator's report |
| [ood_baseline_tasks.csv](ood_baseline_tasks.csv) | All 20 task names, denominators, successes, errors, actions, and costs | Unchanged copy of the aggregator's task table |
| [ood_matched_controls.json](ood_matched_controls.json) | Completed matched fresh-noise 91/200 and reused-noise 90/200 results, paired uncertainty, and episode-source hashes | Unchanged historical control summary, superseded for all-method comparison by the complete paired report |
| [ood_paired.json](ood_paired.json) | Exact paired results: Euler 86/200, RK4 fresh 91/200, RK4 reused 90/200, genuine Astra reversal 5/200; all method/task results, uncertainty, costs, and source hashes | Unchanged copy of the complete outcome aggregator's report |
| [ood_paired_methods.csv](ood_paired_methods.csv), [ood_paired_tasks.csv](ood_paired_tasks.csv) | All four method/suite totals and 80 method/task rows | Unchanged aggregator tables |
| [ood_paired_episodes.csv](ood_paired_episodes.csv) | All 800 episode outcomes, pairing/reset identities, costs and original source-row hashes | Exact selected fields from the canonical episode JSON; no outcome rounding or filtering |
| [ood_runtime_audit.json](ood_runtime_audit.json) | Complete 25-archive/32-shard/200-episode audit: 39/3,108 numerical failures; every failing full/action/padding metric retained, all provider bindings and latent reuse verified | Unchanged final read-only audit summary |
| [ood_archive_provenance.json](ood_archive_provenance.json) | Archive receipts/hashes, per-unit audit hashes, all episode IDs, reset/runtime/config identities, offline validation and the resolved local dependency omission | Compact cross-verified provenance; omission rules documented inside |
| [ood_provider_review.json](ood_provider_review.json) | All recorded calls, schema/controller acceptance, 545 actual fallback actions, disjoint fallback windows and provider-error retry consequences | Unchanged final archive-bound review; no raw provider response bodies |
| [ood_reversal_success_context.json](ood_reversal_success_context.json) | Five successful reversal episodes joined to actual fallback windows; one success contains 20 fallback actions | Unchanged archive-bound outcome/attribution review |
| [runtime_numerics.json](runtime_numerics.json) | All 14 conditions at each of RK4/100, /200, and /500; full error metrics, costs, native parity, source/checkpoint identities, and artifact hashes | Compact review summary of the 1.45 MB numerical gate |
| [astra_proposal_preflight.json](astra_proposal_preflight.json) | Passed genuine Stage 1 proposal round trip, actual model, request/response/provider hashes, controller and full internal errors | Unchanged copy of the GPU preflight report |
| [astra_development_smoke.json](astra_development_smoke.json) | Exact completed development episode summary: unsuccessful task, 26/28 roundtrips passing, zero fallback | Unchanged worker summary |
| [astra_development_review.json](astra_development_review.json) | All 28 recorded roundtrip metrics and independently checked provider/observation bindings; two numerical failures retained | Unchanged read-only review report |
| [astra_development_smoke.mp4](astra_development_smoke.mp4) | Original 211 KB video of the unsuccessful 520-action development episode | Unchanged worker video |
| [astra_development_replay_plan.json](astra_development_replay_plan.json) | Fixed development cases 310/330/485 and resolutions 100/200/500, with exact original array/provider bindings | Input plan recorded before the separate GPU diagnostic |
| [astra_development_replay.json](astra_development_replay.json) | All nine replay pairs, exact N100 reproduction, source/provider/runtime identities, and independently verified array errors; step 330 fails at every resolution | Compact review of the completed worker report and lossless archive; no replacement solver selection |
| [payload_source_parity.json](payload_source_parity.json) | Execution/core commit identities versus frozen payload; exact file hashes and three harmless formatting/import differences | Code-identity review, not an implementation-correctness proof |
| [OOD example video metadata](ood_examples/video_manifest.json) | Same first frozen episode in each suite for all four methods, chosen by episode ID rather than outcome | Original videos; hashes, frame counts, and source paths retained |
| [snapshot_sources.json](snapshot_sources.json) | Source paths, source and snapshot SHA-256 values, sizes, and copy/compaction descriptions | Snapshot provenance manifest |

The standard LIBERO-10 455/500 baseline is already tracked in
[checkpoints/libero_l40s_full_baseline.json](../checkpoints/libero_l40s_full_baseline.json).
The interpretation and protocol limits are in [RESULTS.md](../RESULTS.md) and
[NUMERICAL_RESULTS.md](../NUMERICAL_RESULTS.md). All four paired OOD methods and
all final archive audits are complete. Astra reversal scored 5/200, below both
matched controls, and 39 same-condition roundtrips exceeded the unchanged 0.02
internal-space limit. Its earlier development smoke also failed its task and
two numerical checks. Task success, recording integrity, provider acceptance,
and numerical accuracy are separate results.

The paired report's `valid_complete_evaluation` flag concerns execution checks,
not passing numerical validation. The CSV preserves every episode needed for
paired outcome analysis, including all four methods' identical reset keys.
Original event sequence/timestamps and duplicated absolute paths are omitted;
the complete canonical JSON is bound by SHA-256 in `snapshot_sources.json`.
The final archive provenance retains all per-unit audit identities and compact
counts while the verbose 96,066-array inventories remain in the source audits.

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
- [Complete paired aggregation](../artifacts/ood_paired_report/report.md),
  [full runtime audits](../artifacts/ood_runtime_audit), and
  [provider review source/validation](../artifacts/ood_provider_review).
- [Complete runtime gate](../artifacts/osmo_runtime_l40s/runtime_diagnostics.json).
- [Original proposal preflight](../artifacts/osmo_ood_steering_group0/astra_proposal_diagnostics.json)
  and [original vision-smoke files](../artifacts/astra_vision_smoke).

Public object URLs containing a `generation` query parameter identify pinned
asset versions; they are not credential-bearing signed URLs.

## Fixed episode videos

Each row below uses task 0 / state 0 from one suite, for all four methods.
These examples were selected by episode ID rather than task outcome. They show
simulation frames at 20 fps and omit model/API waiting time; they are not a
latency comparison or a success-rate estimate.

| Suite | Euler fresh | RK4 fresh | RK4 reused | Astra reversal |
|---|---|---|---|---|
| Goal-OOD | [Video](ood_examples/libero_goal_ood/baseline_euler10.mp4) | [Video](ood_examples/libero_goal_ood/policy_fresh.mp4) | [Video](ood_examples/libero_goal_ood/policy_reused.mp4) | [Video](ood_examples/libero_goal_ood/reversal.mp4) |
| Spatial-OOD | [Video](ood_examples/libero_spatial_ood/baseline_euler10.mp4) | [Video](ood_examples/libero_spatial_ood/policy_fresh.mp4) | [Video](ood_examples/libero_spatial_ood/policy_reused.mp4) | [Video](ood_examples/libero_spatial_ood/reversal.mp4) |
