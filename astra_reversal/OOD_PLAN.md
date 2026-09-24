# Frozen LIBERO-OOD evaluation plan — 2026-09-24

The requested benchmark is the 20-task LIBERO-OOD release accompanying
[Li, arXiv:2505.03500v5](https://arxiv.org/abs/2505.03500v5). Use the two
ten-task suites `libero_goal_ood` and `libero_spatial_ood`, ten trials per task,
300 policy action steps, ten stabilization steps, and five executed actions per
policy call. Preserve released prompts, including their spelling, and modified
success predicates. Pin the environment to
`QuanyiLi/pi0-text-latent@587a6cbf64f16c7b87fa5805dc0ed934192239a4`.

Use the release's default seed 7. Freeze each task's ten reset states before
evaluation, including randomized fixture model transforms as well as dynamic
simulation state. Every paired condition replays these same states. Policy noise
uses the existing explicit `SeedSequence([episode_seed, step, stream])` scheme;
this common schedule is repeated across initial states. This differs from an
uninterrupted policy RNG stream and is declared rather than presented as exact
replication of the author's numerical results.

Keep the verified, frozen `lerobot/pi05_libero_base` weights and the explicit
OpenPI LIBERO input profile from the completed standard-suite evaluation.
Checkpoint training overlap with these OOD tasks is unknown. This is a new
pi05 evaluation on the released tasks; the paper's pi0 and TLI results describe
different policies and methods.

The first condition is the Euler-10 fresh-noise baseline. The primary steering
condition is SPEC Stage 1: genuine `azure/openai/gpt-6-astra` receives the two
current RGB views, robot proprioception, controller specification, instruction,
and bounded history. It directly supplies every value in a ten-by-seven action
chunk. It receives no simulator object poses or success predicates. Invert the
normalized, padded proposal with the full task instruction and fixed current
observation; reuse the resulting full noise tensor for policy calls on fresh
observations, refreshing Astra every 20 environment steps or on declared events.
No pi05-generated trajectory may substitute for an Astra proposal without an
explicit fallback record.

Select flow numerics only on saved standard-LIBERO development observations,
before OOD steering. The previous single-observation RK4-50 cubic-grid gate did
not generalize across its development rollout. Validate all 14 recorded
conditions with unchanged noise tolerance 0.1 and action tolerance 0.02; screen
larger step counts by cost and retain all failures. Do not select solver or
prompt settings using OOD success. Report any remaining numerical limitations.

Use synchronous simulation, a JSON proposal contract, one bounded regeneration
after invalid responses, and the existing logged fresh-noise policy fallback.
Record actual endpoint model, request settings, response usage, latency,
accepted/rejected proposals, fallback counts, action clipping, flow evaluations,
and lossless inversion tensors. The initial API settings are low reasoning,
JSON output, no specified temperature, and cache bypass; the full resolved
configuration and prompt source are retained with each run.

Report success per task and suite, paired episode outcomes, execution errors,
and cost. Preserve solver identity when comparing conditions; a change from
Euler to a higher-order solver is a confound unless accompanied by a matched
fresh-noise control. Reused-noise and direct-Astra controls help distinguish
latent reuse from Astra proposal quality. Stage 2 augmentation is a separate
condition, never relabeled as direct numeric Stage 1 steering.

All policy/simulator computation runs on allocated OSMO L40S GPUs. The baseline
and steering group 0 use `groot-l40s-03`; groups 1–3 use independent single-GPU
tasks on `groot-l40s-01` with LOW-priority borrowing. This scheduling adjustment
uses spare four-GPU nodes after pool 03 lacked healthy eight-GPU capacity. Each
worker retains the original logical shard assignment and byte-identical method
configurations, and repeats the numerical and checkpoint gates. Actual worker
identity and per-worker timing are retained. No OOD outcome changes solver,
prompt, reset, or method settings.

Inference credentials are injected from a named OSMO secret and excluded from
source payloads, logs, and result archives.
