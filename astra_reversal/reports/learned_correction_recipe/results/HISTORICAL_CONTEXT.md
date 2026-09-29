# Historical acquisition context

Whole source-study acquisition, not cost attributable to the selected 12 teachers or seven historical native anchors.

| Whole source study | Calls | Input | Output | Total | Missing usage |
|---|---:|---:|---:|---:|---:|
| [Complete seed29 phase study](../../phase_interpolation/evaluation/results/report.json) | 679 | 5,543,789 | 214,185 | 5,757,974 | 0 calls |
| [Interrupted seed19 representation pilot](../../representation_steering/development/results/report.json) | 220 | 1,113,516 | 15,530 | ≥1,129,046 | 122 calls |

These two whole source studies recorded 899 calls and at least 6,887,020 tokens. They include other arms, unsuccessful trials and unused trajectories. The selected twelve teachers' acquisition cost has not been isolated, so this total must not be labeled their exact cost or added to new evaluation tokens. Reasoning tokens are already included in output tokens. The pilot's 122 HTTP429 calls have unknown usage; 48 were explicitly classified as budget-exceeded errors.

The new recipe makes no experiment-time Astra inference API calls. Coding, review and orchestration assistant tokens were not instrumented and are outside this zero-call claim. Its native anchor collection, frozen feature extraction, two optimizer fits and 240 physical evaluation trials are reported as separate new compute stages. No monetary price, amortized saving or superiority to another learning method is inferred.

The historical efficacy cohorts use different resets, assistance budgets and selection rules. They are not pooled with the seed61 unconditional test. The selector imitates successful intervention support on known compositions; its gate is not a calibrated failure probability.

[Method design](../DESIGN_REVIEW.md) · [Current report](index.html) · [Exact context numbers and source hashes](historical_context.json)
