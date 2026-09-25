Iterative intervention evaluation: 20 verified case recordings.

Astra integration: **provider_and_execution_observed**. 316/323 provider proposals accepted; 316 Astra intervention rollouts executed. Shared baseline successes are not evidence of intervention benefit.

Common identity baseline: 8/20. Known-noise control: 8/20. Fresh-noise control: 6/20.

Counts are cumulative successes by attempt; the baseline is attempt 1. Rescue is conditional on baseline failure. Token costs run through first success or cap.

| Arm | 1 | 2 | 3 | 4 | 5 | Median first success¹ | Median rescue revisions | Censored | Rescue | Calls | Input / output / reasoning² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| random_noise | 8 | 8 | 8 | 8 | 10 | 1.0 | 4.0 | 10 | 2/12 | 0 | 0 / 0 / 0 |
| noise_only | 8 | 8 | 9 | 9 | 9 | 1 | 2 | 11 | 1/12 | 46 | 184663 / 13553 / 51 |
| language_only | 8 | 9 | 9 | 9 | 9 | 1 | 1 | 11 | 1/12 | 45 | 172879 / 13561 / 2868 |
| vision_only | 8 | 9 | 9 | 9 | 9 | 1 | 1 | 11 | 1/12 | 45 | 170882 / 14381 / 3285 |
| noise_language | 8 | 9 | 9 | 9 | 9 | 1 | 1 | 11 | 1/12 | 45 | 184187 / 17204 / 2355 |
| noise_vision | 8 | 8 | 8 | 9 | 10 | 1.0 | 3.5 | 10 | 2/12 | 47 | 190990 / 17969 / 2284 |
| language_vision | 8 | 8 | 8 | 9 | 9 | 1 | 3 | 11 | 1/12 | 47 | 184584 / 16980 / 3845 |
| joint | 8 | 8 | 8 | 8 | 8 | 1.0 | — | 12 | 0/12 | 48 | 202388 / 22505 / 4489 |

¹ Among successful cases only. ² Reasoning is included in output tokens. A token sum marked missing is partial, including failed provider calls when usage exists.

Physical execution: 424 rollouts, 121433 actions, 267580 velocity evaluations (24600 for initialization), 323 provider calls. Per-arm standalone costs and paired rescue counts against random search are in report.json and arms.csv.
Physical provider tokens: input 1,290,573; output 116,153; total 1,406,726; reasoning 19,177. Reasoning is included in output tokens.
Physical provider usage is unavailable for 0 of 323 calls; missing usage is not zero cost. budgets.csv separates cumulative search-to-success tokens from actual development-hook tokens.

Report source SHA-256: `e956ad36d68ae752f9414550ec18973b91cc5c507a940eacdcd7b01170c51be8`. Report content digest: `65308e1dd2cedc6f17011c1a26dc9f4d7dfff648f174e3336bf8c8642bfaaf0c`.

- Online adaptation with simulator reset access; not zero-shot evaluation. The OOD tasks and seed7 outcomes were previously observed. Checkpoint training overlap is unknown; seed19 is a follow-up reset condition.
- A rejected proposal consumes an attempt, even when no rollout executes. Unsuccessful searches remain right-censored after attempt 5; success-only iteration medians do not summarize the failures.
- Reasoning tokens are a subset of output tokens and must not be added twice. Token sums with missing records are partial observed costs. No USD price or missing usage is imputed.
- One identity baseline is physically executed per case and attributed to each arm for a standalone comparison. Standalone arm costs must not be summed as physical experiment costs. Development v1 attempts one revision per arm after baseline success; v2 continues rejected proposals until one revision executes, within the same five-attempt cap.
- Initialization uses RK4/100/power3; rollout execution uses Euler/10/power1. Initialization velocity evaluations are separate from rollout evaluations. The baseline's policy and wall times include initialization, recording and archive synchronization; case wall time includes all search work. Parallel worker wall times are retained separately, not pooled into a latency p95.
- Hashes bind the downloaded reports, events, provider ledgers and complete reset manifests. The audit verifies recorded initialization errors and paired reset hashes, not independent numerical reconstruction or native simulator replay. Raw array artifacts remain separate.
- Noise-only Astra proposals versus random noise share the same basis and bounds. Comparisons of language/vision arms against random noise change the intervention operator as well as the proposal source. The 20-case OOD follow-up is exploratory and uses known tasks with new resets.
- Regenerated random directions allow at most four float64 ULPs of coefficient normalization roundoff across CPU libraries; basis identities, coefficient shapes, kinds and scales remain exact. Observed differences are logged, and recorded proposals are never altered.
