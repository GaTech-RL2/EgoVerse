# Full FRS evaluation interrupted by the inference spending cap

The eight-worker L40S run `astra-pi05-frs-evaluation-20260925-1` ended at 2026-09-26T19:22:18.658405 UTC after the inference provider returned `budget_exceeded`. All experiment processes were stopped and their final archives retained. This is an incomplete experiment, not an all 20-task result. The configured key's account-wide spending cap does not identify this study's dollar cost.

There are 5 independently audited, sealed completed tasks out of 20 planned. The final worker summaries contain 952 completed physical rollouts and 197,806 actions in those rollouts. These counts omit unfinished rollout prefixes; the execution-selected task subset is not a representative SR estimate.

The final archived ledgers contain **6,587 calls**, 5,405 accepted and 1,182 failed/rejected. Reported usage is **at least 15,562,019 tokens**, with 1,166 calls missing total usage. There are 1,159 HTTP 429 responses, of which 242 explicitly expose the allowlisted `budget_exceeded` type. Unknown or differently shaped errors are not silently reclassified. Reasoning usage is an output subset.

Every final worker archive was read to gzip EOF and bound by compressed SHA256/size to its retained summaries and provider ledgers. This supports descriptive cost accounting; it is not a full numerical/behavior audit of interrupted cases. [Machine-readable evidence](evaluation_interruption.json) retains all scopes and hashes without credentials or provider aliases.

The completed three-case FRS development results remain unchanged. The [new representation report](../representation_steering/index.html) separately describes TEI/TLI/VEI/VLI and the same account-budget interruption. A funded credential is needed before the remaining Astra comparisons can be completed; no replacement has been launched.

The five fully audited completed tasks also contain 30 candidate judgments: 29 same and 1 uncertain, with zero promotions, auxiliary updates or optimizer steps. This is a learning-gate observation from the interrupted subset, not a full 20-task success estimate. [Bound task evidence](evaluation_gate_observations.json).
