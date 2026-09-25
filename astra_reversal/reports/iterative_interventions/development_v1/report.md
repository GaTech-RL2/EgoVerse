Iterative intervention development: 2 verified case recordings.

Astra integration: **failed**. 0/14 provider proposals accepted; 0 Astra intervention rollouts executed. Shared baseline successes are not evidence of intervention benefit.

Common identity baseline: 2/2. Known-noise control: 2/2. Fresh-noise control: 2/2.

Counts are cumulative successes by attempt; the baseline is attempt 1. Rescue is conditional on baseline failure. Token costs run through first success or cap.

| Arm | 1 | 2 | 3 | 4 | 5 | Median first success¹ | Median rescue revisions | Censored | Rescue | Calls | Input / output / reasoning² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| random_noise | 2 | 2 | 2 | 2 | 2 | 1.0 | — | 0 | 0/0 | 0 | 0 / 0 / 0 |
| noise_only | 2 | 2 | 2 | 2 | 2 | 1.0 | — | 0 | 0/0 | 0 | 0 / 0 / 0 |
| language_only | 2 | 2 | 2 | 2 | 2 | 1.0 | — | 0 | 0/0 | 0 | 0 / 0 / 0 |
| vision_only | 2 | 2 | 2 | 2 | 2 | 1.0 | — | 0 | 0/0 | 0 | 0 / 0 / 0 |
| noise_language | 2 | 2 | 2 | 2 | 2 | 1.0 | — | 0 | 0/0 | 0 | 0 / 0 / 0 |
| noise_vision | 2 | 2 | 2 | 2 | 2 | 1.0 | — | 0 | 0/0 | 0 | 0 / 0 / 0 |
| language_vision | 2 | 2 | 2 | 2 | 2 | 1.0 | — | 0 | 0/0 | 0 | 0 / 0 / 0 |
| joint | 2 | 2 | 2 | 2 | 2 | 1.0 | — | 0 | 0/0 | 0 | 0 / 0 / 0 |

¹ Among successful cases only. ² Reasoning is included in output tokens. A token sum marked missing is partial, including failed provider calls when usage exists.

Physical execution: 8 rollouts, 2454 actions, 7410 velocity evaluations (2460 for initialization), 14 provider calls. Per-arm standalone costs and paired rescue counts against random search are in report.json and arms.csv.
Physical provider tokens: input unknown (14 calls missing usage); output unknown (14 calls missing usage); total unknown (14 calls missing usage); reasoning unknown (14 calls missing usage). Reasoning is included in output tokens.
Physical provider usage is unavailable for 14 of 14 calls; missing usage is not zero cost. budgets.csv separates cumulative search-to-success tokens from actual development-hook tokens.

Report source SHA-256: `82999eb665e155b1aa360cc5caccfe5ed9ba5aeeb6098989cdc7e1e606ee7220`. Report content digest: `59f00e98ccfca53eabbe5024ccfbff22ed3fb69161b627680e09976d5fd0eb91`.

- Online adaptation with simulator reset access; not zero-shot evaluation. The OOD tasks and seed7 outcomes were previously observed. Checkpoint training overlap is unknown; seed19 is a follow-up reset condition.
- A rejected proposal consumes an attempt, even when no rollout executes. Unsuccessful searches remain right-censored after attempt 5; success-only iteration medians do not summarize the failures.
- Reasoning tokens are a subset of output tokens and must not be added twice. Token sums with missing records are partial observed costs. No USD price or missing usage is imputed.
- One identity baseline is physically executed per case and attributed to each arm for a standalone comparison. Standalone arm costs must not be summed as physical experiment costs. Development v1 attempts one revision per arm after baseline success; v2 continues rejected proposals until one revision executes, within the same five-attempt cap.
- Initialization uses RK4/100/power3; rollout execution uses Euler/10/power1. Initialization velocity evaluations are separate from rollout evaluations. The baseline's policy and wall times include initialization, recording and archive synchronization; case wall time includes all search work. Parallel worker wall times are retained separately, not pooled into a latency p95.
- Hashes bind the downloaded reports, events, provider ledgers and complete reset manifests. The audit verifies recorded initialization errors and paired reset hashes, not independent numerical reconstruction or native simulator replay. Raw array artifacts remain separate.
- Noise-only Astra proposals versus random noise share the same basis and bounds. Comparisons of language/vision arms against random noise change the intervention operator as well as the proposal source. The 20-case OOD follow-up is exploratory and uses known tasks with new resets.
