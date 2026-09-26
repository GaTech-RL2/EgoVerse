Image perturbation report: complete

Completed 20/20 cases; independent audits passed for 20. Efficacy released: True.

Recorded completed-case physical work: 167 rollouts, 43,594 actions, 113,140 velocity evaluations; 440 provider calls (1 rejected), at least 11,077,106; 1 calls have unknown totals tokens. Preflight failures: 0.

| Arm | Successes | Baseline wins | Rescues | Censored | Rescue revisions median [min,max] | Rescue tokens median [min,max] |
|---|---:|---:|---:|---:|---|---|
| random_noise | 10/20 | 8 | 2 | 10 | 1.0 [1,1] | 0.0 [0,0] |
| random_occlusion | 10/20 | 8 | 2 | 10 | 1.0 [1,1] | 0.0 [0,0] |
| random_demo_blend | 11/20 | 8 | 3 | 9 | 1 [1,2] | 0 [0,0] |
| astra_occlusion | 12/20 | 8 | 4 | 8 | 1.0 [1,1] | 110780.5 [87442,283209] |
| astra_demo_blend | 12/20 | 8 | 4 | 8 | 1.0 [1,2] | 112073.0 [88801,387762] |

Matched comparisons use every failed-baseline pair, including capped failures:

- astra_occlusion vs random_occlusion: both rescue 2, Astra only 2, random only 0, neither 8.
- astra_demo_blend vs random_demo_blend: both rescue 3, Astra only 1, random only 0, neither 8.

- Exploratory adaptation on known task compositions with recorded resets; this is not a held-out-task or zero-shot claim.
- All arms share one recovered-noise baseline. A baseline success is not an intervention rescue. A failed capped search remains in the outcome denominator.
- The RGB operators keep task language and recovered noise fixed. A successful trajectory can include native fallback actions or accepted no-op decisions; acceptance is not proof of a nonzero effect or causation.
- The fixed training-donor catalog is not selected from evaluation outcomes. Training overlap with the policy checkpoint is not established as absent.
- Rescue-only timing/iteration/token statistics condition on a failed baseline and later success. Censored cases are listed separately and receive no invented time-to-success.
- Physical costs count the shared baseline, initialization and image gate once. Standalone arm costs repeat common setup and must not be added across arms.
- Physical totals cover the reconciled completed cases only. Donor preparation, transport diagnostics and any interrupted or incomplete execution require separate overhead receipts.
- Every physical provider call, including rejection, contributes available usage. Missing usage stays unknown; reasoning tokens are a subset of output, and no dollar price is assumed.
- Summed rollout/case wall time is measured work across cases, not elapsed workflow time; rollout time includes inference, recording and baseline setup.
- This reporter reconciles summaries and audit receipts. The bound independent audits verify recorded arrays, feedback and provider provenance without replaying simulator physics or hidden model states.
- Development may force one extra rollout after baseline success. These physical costs do not change zero intervention tokens to that baseline success.
