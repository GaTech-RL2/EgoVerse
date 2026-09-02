# UNITE U-Socket register sweep — pre-smoke launch handoff

Date: 2026-09-02
Status: **SMOKE_READY / SMOKE_REQUIRED / FULL LAUNCH BLOCKED**

## Correct architecture contract

This is an action-token-to-latent-register sweep. It is not an observation
patch-token sweep and it is not an exact replication of the source UNITE image
model.

```text
clean normalized action sequence A: [B, 16, 4]
  -> tokenization input: 16 clean action tokens
  -> generative encoder
  -> latent registers Z: [B, N, 16], N in {4, 8}
  -> action decoder
  -> reconstructed normalized action sequence: [B, 16, 4]

observation context C:
  image + proprio -> FusedObsEncoder -> one pooled observation embedding
  -> AdaLN conditioning at every denoiser block
  -> 32 repeated learned-position in-context tokens inserted at block 4
     in the same self-attention stream
```

The clean action tokens take the architectural role occupied by clean image
patches in the source tokenization/reconstruction path. The pooled observation
does not enter tokenization, is not reconstructed, and is not a sweep variable.
During denoising it drives AdaLN at every block and is repeated into 32
learned-position, nonspatial in-context tokens at block 4. These are ordinary
self-attention tokens, not observation patches and not cross-attention keys.
The implementation field `num_latent_tokens` means **latent register count**
in this adaptation.

The other sweep axis is the Generative Encoder topology:

- `shared`: tokenization and denoising reuse the same Generative Encoder
  backbone.
- `separate`: tokenization and denoising use distinct Generative Encoder
  backbones.

## Only active config set

Exactly four materialized row configs are active. There is no active generic
base model or generic sweep experiment config. Do not create or select any
additional per-row YAML.

Active row configs:

| Row ID / model config | GE topology | Register count | Required overrides |
|---|---|---:|---|
| `us_unite_register_shared_nt4_s42` | shared | 4 | `share_encoder_denoiser=true`, `num_latent_tokens=4` |
| `us_unite_register_shared_nt8_s42` | shared | 8 | `share_encoder_denoiser=true`, `num_latent_tokens=8` |
| `us_unite_register_separate_nt4_s42` | separate | 4 | `share_encoder_denoiser=false`, `num_latent_tokens=4` |
| `us_unite_register_separate_nt8_s42` | separate | 8 | `share_encoder_denoiser=false`, `num_latent_tokens=8` |

Each row lives at
`egomimic/hydra_configs/model/bf/<row-id>.yaml`. These are the only row model
configs in the intended active set. Their `nt` spelling is the implementation
field name; in all scientific and collaborator-facing descriptions it means
latent-register count, never observation-token or patch-token count.

The two former N=16 row configs were removed from the Hydra model-config tree
because that arm is not requested. They remain recoverable from the parent Git
commit and the dated pre-cleanup archive, but cannot be selected accidentally
from this branch.

The manifest is the sole row authority. No patch-token arm, observation-token
count arm, or old parameter-count table belongs to this active set.

## Joint-update-safe gradient telemetry

All four active rows preserve the intended joint-update objective: every optimizer step
uses the reconstruction loss together with the aggregate flow loss. Their
wrapper controls are therefore:

```yaml
unite_flow_updates_per_reconstruction: 0
unite_gradient_telemetry_every_n_steps: 100
```

The released wrapper now returns exactly reconstruction plus flow as the
optimizer loss and uses read-only `autograd.grad` calls on that same forward
graph at telemetry cadence. These calls do not populate or modify
`parameter.grad`. Shared rows emit finite cosine and component norms; separate
rows emit disjoint tokenizer-reconstruction and denoiser-flow norms without
fabricating a cosine. The two-step smoke uses cadence 2 so telemetry is measured
after the first optimizer update. A real two-rank optimizer-plus-validation
smoke is still required before a full run.

## Corrected artifacts

- Sweep manifest:
  `unite_usocket_register_sweep_manifest.yaml`
  - SHA-256:
    `75a7a081734ae585c0672399caaeb9913f03e1564f150b0fcffe5f385d482f19`
  - Gate state: `artifact_status=SMOKE_READY`,
    `launch_status=SMOKE_REQUIRED`
- Four-active-row train/rollout graph artifact:
  `artifacts/unite_register_sweep_20260902/config_graphs/four_active_rows_train_rollout.json`
  - SHA-256:
    `c850d6146fb6bf9bc84e7332f9988d9bd446f34f89c291f4d7670f271fa29e20`
  - Lint result: `8/8` train/rollout graphs clean.
- Interactive graph:
  `docs/config_graphs/unite_register_sweep_20260902/us-unite-register-sweep.html`
  - SHA-256:
    `0e7bd82a277952a366b737852ac7405ad86b2b13320aef3efa918153cc748a7b`
  - Its adjacent JSON mirror is byte-identical to the canonical graph artifact.

All four active rows now have construction-derived, byte-canonical durable
parameter manifests. `total` equals `trainable` in every row.

| Active row | Total/trainable parameters | GE backbone parameters | Durable manifest | SHA-256 |
|---|---:|---:|---|---|
| `us_unite_register_shared_nt4_s42` | 226,292,820 | 129,999,376 | `artifacts/unite_register_sweep_20260902/parameter_manifests/us_unite_register_shared_nt4_s42.json` | `4f5b58f0e8174a3faf43c86aef4746c3531ef423d86d4efac9978a84bc45466a` |
| `us_unite_register_shared_nt8_s42` | 226,295,892 | 130,002,448 | `artifacts/unite_register_sweep_20260902/parameter_manifests/us_unite_register_shared_nt8_s42.json` | `b73d91410658f0d4cf4c0e6aee378471dfc61ddaaba1a8346d67b9f19354027d` |
| `us_unite_register_separate_nt4_s42` | 356,308,996 | 259,998,752 | `artifacts/unite_register_sweep_20260902/parameter_manifests/us_unite_register_separate_nt4_s42.json` | `939ffee43c81e31792d838b73a4c8e01086e1dd0b5114932e435cb24c957b795` |
| `us_unite_register_separate_nt8_s42` | 356,315,140 | 260,004,896 | `artifacts/unite_register_sweep_20260902/parameter_manifests/us_unite_register_separate_nt8_s42.json` | `dd9a59125d5ad2eec5721019c93cee9ca509e0f5bf29ec37ea1648d79a1e367b` |

## Stale and historical configuration policy

- Patch-named or observation-patch-sweep artifacts are not active inputs and
  must not be used as inputs for this sweep.
- Do not delete historical tracked configs as part of this WIP correction.
- Do not treat unrelated UNITE, temporal-codec, AdaLN, or SpatialSoftmax
  experiments as alternate rows of this sweep.
- The inactive generic model and experiment YAMLs were removed after their test
  references were migrated. The complete pre-correction loose draft was
  archived at
  `/coc/flash7/paphiwetsa3/backups/unite_usocket_released_draft_pre_register_cleanup_20260902.tar.gz`
  (SHA-256
  `b078758f6ff6cdd5eb86985401114d8049ad417dee8dd8cc0b28cdbc24a7ff85`).
- The maintained external
  `/coc/flash7/paphiwetsa3/scripts/train/flow_transfer_unite_skynet_x2_v20.sbatch`
  now consumes schema 2, selects the four active row configs directly, binds
  `ddp_find_unused_parameters_true`, and contains no `obs33tok` or
  CrossTransformer tag. Each row binds a durable repo parameter manifest under
  `artifacts/unite_register_sweep_20260902/parameter_manifests/`; the launcher
  SHA-checks it, regenerates the resolved-stage payload, requires byte-identical
  content, and preserves both copies in run provenance. Its SHA-256 is
  `1c8a0ffcf144b1e13dbe7040518255709e7dd27af271aaa6492084be2ee040bd`.
  The preceding pre-readiness-identity version is archived at
  `/coc/flash7/paphiwetsa3/backups/flow_transfer_unite_skynet_x2_v20.pre_readiness_identity_20260902.sbatch`.
  Its pre-schema-2 version is archived at
  `/coc/flash7/paphiwetsa3/backups/flow_transfer_unite_skynet_x2_v20.pre_schema2_20260902.sbatch`.
- After the schema-2 launcher was frozen, a concurrent stale schema-1
  patch-token copy temporarily replaced the external file. That exact raced
  copy is preserved at
  `/coc/flash7/paphiwetsa3/backups/flow_transfer_unite_skynet_x2_v20.overwritten_after_f04afb_20260902T020550Z.sbatch`
  (SHA-256
  `2e5dc7e8c5039be21339573b6eda4627cc67620da8853969afad15f93c01d8ee`).
  No stale content was merged. The canonical external launcher was restored
  byte-for-byte to its guarded schema-2 version, then advanced deliberately to
  SHA-256 `1c8a0ffcf144b1e13dbe7040518255709e7dd27af271aaa6492084be2ee040bd`
  with a fail-closed smoke-to-full source-identity check.
- One support experiment,
  `egomimic/hydra_configs/experiment/pusht/unite_usocket_register_sweep_val01_h16.yaml`,
  composes the U-Socket-only data and evaluator contract. Row selection remains
  direct; there are no per-row experiment aliases.
- The support experiment enables EnergyScore@32 with seed-bank SHA-256
  `88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6`.
  Model autocast remains BF16, while adaptive DOPRI5 state, derivative, and
  error-control arithmetic remain FP32.
- The launcher verifies the U-Socket subsection of the canonical combined split
  artifact (SHA-256 `672f0f519bb7bff5b6b956d1b709abf1a1d387dd6b88b20a7c37536799bce0cd`)
  and embodiment-19 subsection of the train-only normalization artifact
  (SHA-256 `3559aca1ac1279cbdd37de8e5b2da9bb350fbc0f1177d4c669aa590011fd0203`).

## Why launch remains blocked

The static gate is complete. The only remaining launch blocker is the required
real two-rank optimizer-plus-scheduled-validation smoke for each active row,
including checkpoint/EMA strict reload, finite joint/topology/EnergyScore@32
metrics from both ranks, and W&B visibility. The canonical launcher permits
`MODE=smoke` at `SMOKE_READY / SMOKE_REQUIRED` but rejects `MODE=full` until
row-specific smoke evidence is recorded and the manifest is advanced to
`LAUNCH_READY / READY`. A full run must name the exact passing smoke commit.
The launcher accepts a later readiness commit only when that smoke commit is an
ancestor, the net changed paths are limited to this manifest and handoff, and
the parsed manifest is byte-semantically identical after normalizing only the
two readiness statuses and removal of the sole smoke blocker.

## Access and actions in this correction

`sky1` authenticated but did not execute a bounded trivial command, so the
Skynet access procedure required the `sky2` fallback. The current combined
review recorded:

- Python syntax/import checks, Ruff, and YAML parsing.
- `108/108` in-scope tests across the released-policy, fidelity,
  training-entry, telemetry, launcher, and normalization-path suites. These
  include real H16 clean-action
  materialized tokenize/denoise/backward coverage for shared/separate x
  `N={4,8}`,
  configured-wrapper selection, Muon/AdamW grouping, optimizer stepping, and
  optimizer-state round-trip.
- Canonical config-graph lint for all four selectable rows in train and rollout
  modes (`8/8` graphs clean), recording `max_content_tokens=16`,
  `max_condition_tokens=1`, and enabled GE gradient checkpointing.
- The canonical trainer now honors `model._target_`; the released wrapper uses
  stable model-local parameter names; and the Torch 2.7.1 runtime uses a
  provenance-pinned backport of the official PyTorch Muon implementation.

The launch-plumbing follow-up added topology-aware joint-update telemetry,
schema-2 launcher parsing, the U-only EnergyScore support experiment, combined
split/norm subsection checks, durable parameter-manifest byte-identity checks,
optimizer-group assertions, and the released-smoke verifier path. Its focused
telemetry/launcher suite passed 12/12 tests, including real shared/separate
policy forward-optimizer-forward telemetry checks. Ruff, Python compile,
`bash -n`, YAML parsing, graph lint, and `git diff --check` passed.

The first real four-row smoke attempt was Slurm array `3741178`. All rows were
stopped before their first optimizer step because
`ReleasedUniteModelWrapper.configure_optimizers()` incorrectly called
`named_parameters()` on `PipelineAlgo`, which intentionally is not an
`nn.Module`. Consequently, this attempt produced no qualifying smoke evidence,
checkpoint, or strict checkpoint/EMA reload; its short-lived W&B runs are failed
attempt records only. The wrapper now enumerates Lightning's registered
`self.nets` module tree with stable `nets.policy...` names and duplicate removal.
A real-`PipelineAlgo` regression verifies exact parameter identity coverage,
disjoint AdamW/Muon groups, and the required content/action projection routing.
The complete in-scope suite passes `108/108` after this fix. The manifest remains
`SMOKE_READY / SMOKE_REQUIRED`; all four real smokes must be retried from the
post-fix commit before any full launch.
