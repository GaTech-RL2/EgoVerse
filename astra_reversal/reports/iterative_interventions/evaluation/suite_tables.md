Pooled and per-suite intervention follow-up results.

Success counts include the common identity baseline at attempt 1. Rescue medians exclude baseline successes.

pooled: 20 cases; identity baseline 8/20.

| Arm | Success 1 / 2 / 3 / 4 / 5 | Censored | Rescue | Median rescue revisions | Median rescue tokens | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random_noise | 8 / 8 / 8 / 8 / 10 | 10 | 2/12 | 4.0 | 0.0 | 0 |
| noise_only | 8 / 8 / 9 / 9 / 9 | 11 | 1/12 | 2 | 7904 | 46 |
| language_only | 8 / 9 / 9 / 9 / 9 | 11 | 1/12 | 1 | 3657 | 45 |
| vision_only | 8 / 9 / 9 / 9 / 9 | 11 | 1/12 | 1 | 3587 | 45 |
| noise_language | 8 / 9 / 9 / 9 / 9 | 11 | 1/12 | 1 | 3841 | 45 |
| noise_vision | 8 / 8 / 8 / 9 / 10 | 10 | 2/12 | 3.5 | 15395.5 | 47 |
| language_vision | 8 / 8 / 8 / 9 / 9 | 11 | 1/12 | 3 | 12313 | 47 |
| joint | 8 / 8 / 8 / 8 / 8 | 12 | 0/12 | — | — | 48 |

Unique physical execution: 424 rollouts, 121433 actions, 267580 velocity evaluations, 323 provider calls.

libero_goal_ood: 10 cases; identity baseline 5/10.

| Arm | Success 1 / 2 / 3 / 4 / 5 | Censored | Rescue | Median rescue revisions | Median rescue tokens | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random_noise | 5 / 5 / 5 / 5 / 6 | 4 | 1/5 | 4 | 0 | 0 |
| noise_only | 5 / 5 / 5 / 5 / 5 | 5 | 0/5 | — | — | 20 |
| language_only | 5 / 6 / 6 / 6 / 6 | 4 | 1/5 | 1 | 3657 | 17 |
| vision_only | 5 / 5 / 5 / 5 / 5 | 5 | 0/5 | — | — | 20 |
| noise_language | 5 / 6 / 6 / 6 / 6 | 4 | 1/5 | 1 | 3841 | 17 |
| noise_vision | 5 / 5 / 5 / 5 / 5 | 5 | 0/5 | — | — | 20 |
| language_vision | 5 / 5 / 5 / 6 / 6 | 4 | 1/5 | 3 | 12313 | 19 |
| joint | 5 / 5 / 5 / 5 / 5 | 5 | 0/5 | — | — | 20 |

Unique physical execution: 180 rollouts, 51008 actions, 114380 velocity evaluations, 133 provider calls.

libero_spatial_ood: 10 cases; identity baseline 3/10.

| Arm | Success 1 / 2 / 3 / 4 / 5 | Censored | Rescue | Median rescue revisions | Median rescue tokens | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random_noise | 3 / 3 / 3 / 3 / 4 | 6 | 1/7 | 4 | 0 | 0 |
| noise_only | 3 / 3 / 4 / 4 / 4 | 6 | 1/7 | 2 | 7904 | 26 |
| language_only | 3 / 3 / 3 / 3 / 3 | 7 | 0/7 | — | — | 28 |
| vision_only | 3 / 4 / 4 / 4 / 4 | 6 | 1/7 | 1 | 3587 | 25 |
| noise_language | 3 / 3 / 3 / 3 / 3 | 7 | 0/7 | — | — | 28 |
| noise_vision | 3 / 3 / 3 / 4 / 5 | 5 | 2/7 | 3.5 | 15395.5 | 27 |
| language_vision | 3 / 3 / 3 / 3 / 3 | 7 | 0/7 | — | — | 28 |
| joint | 3 / 3 / 3 / 3 / 3 | 7 | 0/7 | — | — | 28 |

Unique physical execution: 244 rollouts, 70425 actions, 153200 velocity evaluations, 190 provider calls.

- Every suite retains all ten fixed seed19 cases; pooled results retain all twenty.
- Rescue is conditional on common identity-baseline failure. Iteration counts include rejected API proposals.
- A primary rescued-case token median is unknown if any rescued case has missing usage for that field; complete-case-only medians are labeled separately.
- Reasoning tokens are a subset of output tokens. No USD prices or missing token usage are imputed.
- Only noise_only versus random_noise holds the intervention operator/basis fixed; other comparisons also change the operator.
