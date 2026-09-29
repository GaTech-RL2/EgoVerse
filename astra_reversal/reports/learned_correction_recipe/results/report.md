# Learned correction recipe

Status: **complete**. Seed 61; new resets of known compositions. No Astra calls during training or evaluation.

All 240 physical trials passed the recorded-data checks: 40 OOD cases and eight standard-panel cases per arm. Every arm was evaluated independently on every reset.

| Cohort | Method | Success | Gains / harm vs native | Capped failures | Median actions, successes only |
|---|---|---:|---:|---:|---:|
| ood | native | 15/40 | — | 25 | 95 |
| ood | recorded_schedule | 27/40 | 12 / 0 | 13 | 99 |
| ood | learned_selector | 19/40 | 5 / 1 | 21 | 99 |
| ood | flow_head | 14/40 | 2 / 3 | 26 | 101.0 |
| ood | gated_flow_head | 16/40 | 2 / 1 | 24 | 97.0 |
| id_panel | native | 8/8 | — | 0 | 248.0 |
| id_panel | recorded_schedule | 8/8 | 0 / 0 | 0 | 248.0 |
| id_panel | learned_selector | 8/8 | 0 / 0 | 0 | 245.0 |
| id_panel | flow_head | 8/8 | 0 / 0 | 0 | 250.0 |
| id_panel | gated_flow_head | 8/8 | 0 / 0 | 0 | 248.0 |

## Physical cost

Verified evaluation: 240 rollouts; 52,431 actions; 105,620 velocity evaluations including parity probes. These totals are complete.

Training: four native anchor collection rollouts; 601 frozen feature windows; 10,818 feature-extraction velocity evaluations; 1,000 selector updates and 1,000 head updates. New provider calls/tokens: 0/0. Historical teacher acquisition is separate and was not free.

## Interpretation and audit scope

- Training uses successful historical trajectories on known task compositions; new resets do not establish held-out-task or zero-shot generalization.
- The ID retention panel covers four of the ten standard LIBERO-10 tasks, with two prescribed states each.
- Every arm runs unconditionally on every evaluation reset. Successes are not inherited from native and no best-of-attempts selection is used.
- The selector gate imitates successful teacher intervention labels; it is not a calibrated failure probability. Its fixed threshold is strictly greater than 0.5.
- The gated head preserves original native conditioning; it does not compose TEI/TLI with the residual head.
- Executed controller prefixes supervise the head; a native-generated unexecuted suffix supplies input context, not demonstration labels.
- Zero new provider calls excludes the nonzero historical cost of acquiring and auditing teachers. No monetary price or amortized saving is inferred.
- This report verifies immutable recorded metadata, event joins, keyed noise descriptors and paired reset hashes. It does not independently recompute hidden states, optimizer updates or simulator outcomes.
- Head condition IDs are checked against the exact native-ID/head/config composition. Matching raw observation descriptors and prompts must share a native condition ID; non-overlapping observations are not independently rehashed from omitted array bytes.
- Rollout wall time includes reset, policy, simulator and video work; its components must not be added to that wall time. Parallel worker sums are resource time, not experiment elapsed time.
- Videos are recorded external-camera frames before each executed action at 20 fps. They omit inference pauses and the terminal post-action image.

[Machine report](report.json) · [Method table](metrics.csv) · [Every physical trial](rollouts.csv) · [Per-task paired comparisons](per_task.json) · [Input hash inventory](input_provenance.json) · [Protocol](protocol.json) · [Reporter source](source/recipe_report.py.txt) · [HTML](index.html)
