# Iterative intervention measurements

These reports measure the new propose–rollout–feedback–revise experiment. They
are separate from the earlier Stage 1 numeric-action reversal results. The
[frozen protocol](../../INTERVENTIONS.md) defines the operators, paired resets,
five-attempt limit, and cost accounting.

Development is complete. It verifies integration; both development baselines
already succeeded, so it does not measure rescuing a failed policy. The 20-case
Goal-OOD/Spatial-OOD follow-up was submitted as
`astra-pi05-interventions-eval-20260924-1` with the exact development-v2 payload.

| Development run | Actual Astra calls | Accepted / executed Astra candidates | Recorded Astra tokens | Physical rollouts | Actions | Velocity evaluations |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| [v1: failed transport](development_v1/report.md) | 14 | 0 / 0 | Unknown for all 14 calls | 8 | 2,454 | 7,410 |
| [v2: integration passed](development_v2/report.md) | 14 | 14 / 14 | 48,463 input + 3,139 output = **51,602** total | 22 | 7,851 | 18,280 |

The first run received HTTP 400 for every Astra request. The gateway required
the JSON output instruction in the user content as well as the system prompt.
That transport fix and stronger development validation were committed before
the second payload was built. Neither run is omitted from cost accounting.

All 14 accepted development-v2 Astra proposals chose zero edits and reproduced
their successful baselines. They exercised the actual request/execution path,
but demonstrate no nonzero Astra benefit. The separate fixed embedding probes
changed the weighted model's full output by maximum absolute values 0.00449219
and 0.00427425. They performed no simulator actions and are not Astra proposals.
Native Euler and zero-edit parity were exact on both allocated L40S GPUs.

The generated development tables report zero search tokens to first success
because their common baselines succeeded at attempt 1. **The forced integration
trials still cost 51,602 tokens**, included in the physical totals and
`budgets.csv`'s actual-cost columns. Reasoning tokens were reported as zero and
are a subset of output tokens. No endpoint dollar price is assumed.

Two local [transport diagnostics](transport_diagnostics.json) were outside the
rollout experiments: one HTTP 400 call with unavailable usage, and one successful
post-fix call costing 3,487 input + 227 output = 3,714 tokens. Neither executed a
rollout or supplied an experimental candidate. A preceding remote diagnostic
terminated without an observed result, so whether its API call executed remains
unknown. Observed development-plus-diagnostic usage is therefore at least
55,316 tokens, with 15 observed calls missing usage and that remote attempt
unresolved; it is not an invoice total.

Each run includes JSON and CSV records for arms, cases, attempts, and cumulative
budgets. They distinguish first successful attempt, intervention revisions,
censored failures, rescue-only statistics, shared-baseline standalone cost,
and unique physical cost. [Snapshot provenance](snapshot_sources.json) binds
the unchanged report copies to their source bytes and immutable payloads.
The report audit checks events, provider records, resets, and recorded numerical
gates; independent tensor reconstruction is a separate audit.

The [worker-1 independent review](development_v2/worker1_final_read_only_review.json)
verified all 5,627 referenced arrays and 914 recorded generations. Its
[initialization audit](development_v2/worker1_initialization_read_only_audit.json)
recomputed the nonzero fixed embedding probe's effect, including a maximum
0.00369541 change in the seven decoded controller channels. These read-only
checks used saved arrays and made no additional model or simulator calls.
