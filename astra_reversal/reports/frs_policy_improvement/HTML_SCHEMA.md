# Standalone FRS report contract

The renderer consumes an explicit evidence table. It does not run a model, read provider credentials, collect experiments, or infer an unrecorded outcome. The input is an aggregate assembled after the separate recording audit. Supplying an audit receipt hash identifies that evidence; the renderer does not reopen or independently reproduce the receipt.

```sh
source /Users/rpunamiya/Desktop/GEAR/EgoVerse-dataset-weighting-20260919/emimic/bin/activate
PYTHONPATH=. python -m astra_reversal.frs_html_report REPORT.json --output NEW_DIRECTORY
```

The destination must not exist. Outputs are a self-contained `index.html`, exact unchanged input `report.json`, derived numerical tables in `derived.json`, standalone SVG figures, and a file/hash manifest. CSS and SVG are inline in the HTML; no JavaScript, remote fonts, inference, or image generation is used. Source links remain links and must be public HTTP(S) URLs or paths relative to the output directory.

The schema version is `frs-policy-report-1.0`. JSON duplicate keys, nonfinite numbers, inconsistent arithmetic, missing completed coverage, and ambiguous reuse are rejected before publication. Tests use explicitly synthetic fixtures; they are not experiment results.

## Top-level fields

| Field | Content |
|---|---|
| `schema_version` | `frs-policy-report-1.0` |
| `title` | Nonempty display title |
| `status` | `planned`, `partial`, or `complete` |
| `phase` | `development` or `evaluation`; never pool the phases |
| `tasks` | All 20 exact released `{task_key, instruction}` records; keys are `libero_goal_ood:0` through `:9` and `libero_spatial_ood:0` through `:9` |
| `protocol` | Full frozen protocol JSON; at least `seed`, `adaptation_state`, `evaluation_states`, `rounds`, `evaluation_methods`, and `adaptation_methods` |
| `methods` | Unique `{id, label, description}` records, including every referenced method |
| `prompts` | Unique `{id, text, sha256, scope}` records; `sha256` is over exact UTF-8 `text` |
| `notes` | Distinct nonempty plain-text scope/limitation statements |
| `sources` | `{label, href, sha256?}` records |
| `cohorts` | Explicit outcome cohorts described below |
| `overheads` | Physical cost entries outside rollout rows, described below |

Use empty arrays for prompts, cohorts, and overheads in a genuinely planned report. A complete study with provider calls requires its exact prompt text. Plain text is escaped rather than treated as HTML. All 20 task names remain visible in complete task tables, including development tasks outside the selected cohort.

## Cohorts and rounds

Each cohort has `id`, `label`, `status`, `expected_episode_ids`, `method_ids`, `curve_kind`, `rounds`, and optionally `historical: true`.

Episode IDs include reset provenance: `libero_goal_ood:seed43:task0:state1`. For a new evaluation-phase report, the renderer constructs the exact prescribed grid from the full 20-task inventory and the supplied frozen protocol. A checkpoint cohort must cover every evaluation reset and exactly `protocol.evaluation_methods`; an adaptation cohort must cover the adaptation reset and exactly `protocol.adaptation_methods`. A development-phase cohort explicitly declares its smaller denominator. `historical: true` is reserved for an already completed, separately labeled older cohort and bypasses the current reset/method plan; it must still pass its own declared coverage and audit checks.

`curve_kind` has two meanings:

- `checkpoint_evaluation`: each point is the success of that round's checkpoint on the evaluation resets. Success is **not** accumulated across checkpoints. The learned method must have a point after each configured adaptation round. Static comparators may appear only at the final round; if shown elsewhere, their reused physical recordings must be explicit.
- `best_of_attempts`: round 0 is the common baseline, followed by every fixed adaptation revision. Every arm has every prescribed episode in every round, including after earlier simulator success. Curves show cumulative success with simulator reset access; they are not learned-policy success rates.

A round is `{index, label, method_ids, episodes}`. Indices increase without duplicates. Retry indices are contiguous from 0 through the configured cap. Checkpoint rounds can declare a method subset, but the cohort must contain all declared methods and every required learned checkpoint by completion. Each completed round requires exactly one row for each selected method × expected episode pair. There is no failure-only exclusion or fallback denominator.

Pending reports show coverage and recorded physical costs, but suppress new efficacy tables, rescue counts, and curves even if one subcohort has completed. An explicitly historical completed cohort can be displayed separately. A final report requires every declared cohort complete, no execution errors, and passed row audits.

## Episode evidence rows

```json
{
  "episode_id": "libero_goal_ood:seed43:task0:state1",
  "method_id": "native_euler10",
  "physical_run_id": "unique-frozen-workflow-and-rollout-id",
  "reused": false,
  "success": false,
  "error": null,
  "actions": 300,
  "velocity_evaluations": 300,
  "wall_seconds": 0.0,
  "policy_sha256": "<64 lower-case hex characters>",
  "source_sha256": "<exact source recording SHA256>",
  "audit": {"status": "passed", "sha256": "<exact audit receipt SHA256>"},
  "provider_usage": "<usage object below>"
}
```

This is a structural example, not a measured rollout. Numeric fields must contain their actual values. `policy_sha256` identifies the evaluated base or learned checkpoint; `source_sha256` and the audit hash bind the original recording. Pending/failed audits can occur only in unfinished cohorts. A credited success must have at least one executed action and no execution error.

The first occurrence of a physical run has `reused: false`. Every later reference has `reused: true` and identical evidence/cost fields; only the logical `method_id`, reuse flag, and optional notes may differ. Reuse cannot change outcome, policy, provider usage, or source/audit identity. This permits a shared baseline and explicitly repeated static-comparator display without multiplying actual cost. The producer remains responsible for mapping a physical recording to the appropriate logical comparator.

## Provider usage and overheads

The compact usage object has these exact keys:

```json
{
  "calls": 0,
  "accepted_calls": 0,
  "failed_calls": 0,
  "preflight_failures": 0,
  "tokens": {
    "input_tokens": {"sum": 0, "missing_calls": 0},
    "output_tokens": {"sum": 0, "missing_calls": 0},
    "total_tokens": {"sum": 0, "missing_calls": 0},
    "reasoning_tokens": {"sum": 0, "missing_calls": 0}
  }
}
```

All counts are nonnegative integers. `calls` means actual provider calls, including failed/rejected calls; `accepted_calls + failed_calls == calls`. Preflight failures have no network call and are tracked separately. Token sums include every reported physical call, successful or failed. Missing usage is counted per field, never converted to zero. Complete total tokens must equal input plus output; reasoning tokens are a subset of output, not an added cost. No monetary price is inferred.

Each overhead is `{id, label, provider_usage, velocity_evaluations, training_steps, wall_seconds, source_sha256, attribution?}`. Use overheads for setup probes, critiques, comparisons, and fitting that are not already included in rollout rows. Never record the same cost both in a row and an overhead. An optional attribution is exactly `{cohort_id, episode_id, method_id, round_index}`; it connects critiques/judgments to a particular arm and revision. Unattributed global setup remains in physical totals.

Physical totals count each unique rollout and each overhead once. Per-method tables separately show attributed recorded costs, including all capped failures; shared baseline costs can appear in multiple logical method tables. Conditional rescue medians exclude baseline successes and censored failures. Tokens through rescue include all preceding calls and attributed overheads, including failed calls. Rescues with missing usage are counted separately and excluded from the median with complete token accounting. A displayed median must always be read with its rescue and censor denominators.

Summed run/overhead seconds are a work total, not parallel elapsed time or a pooled latency percentile. Timing fields must not overlap; the producer must avoid adding setup wall time twice if it is already inside a rollout timer.

## Limits of the renderer

The supplied task/reset/method grid, row arithmetic, prompt bytes, and reuse identities are checked. Independent full-array, provider, reset, controller, and accepted-replay audits remain the authority for the underlying evidence. The renderer neither recreates simulator predicates nor judges semantic success. Known-task reset separation, training-overlap uncertainty, and absent matched RL comparisons must remain explicit notes. A figure cannot turn an incomplete study or selected example into an efficacy claim.
