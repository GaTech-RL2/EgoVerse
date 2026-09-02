# UNITE U-Socket register sweep — WIP handoff

Date: 2026-09-02
Status: **WIP / NOT LAUNCH-READY / DO NOT SUBMIT**

## Correct architecture contract

This is an action-token-to-latent-register sweep. It is not an observation
patch-token sweep and it is not an exact replication of the source UNITE image
model.

```text
clean normalized action sequence A: [B, 16, 4]
  -> tokenization input: 16 clean action tokens
  -> generative encoder
  -> latent registers Z: [B, N, 16], N in {4, 8, 16}
  -> action decoder
  -> reconstructed normalized action sequence: [B, 16, 4]

observation context C:
  image + proprio -> FusedObsEncoder -> one pooled observation embedding
  -> AdaLN denoising condition only
```

The clean action tokens take the architectural role occupied by clean image
patches in the source tokenization/reconstruction path. Observation has no
declared token geometry in this sweep: its pooled embedding does not enter
tokenization, is not reconstructed, and is not a sweep variable. The
implementation field `num_latent_tokens` means **latent register count** in this
adaptation.

The other sweep axis is the Generative Encoder topology:

- `shared`: tokenization and denoising reuse the same Generative Encoder
  backbone.
- `separate`: tokenization and denoising use distinct Generative Encoder
  backbones.

## Only active config set

Exactly six materialized row configs are active. There is no active generic
base model or generic sweep experiment config. Do not create or select any
additional per-row YAML.

Six row configs:

| Row ID / model config | GE topology | Register count | Required overrides |
|---|---|---:|---|
| `us_unite_register_shared_nt4_s42` | shared | 4 | `share_encoder_denoiser=true`, `num_latent_tokens=4` |
| `us_unite_register_shared_nt8_s42` | shared | 8 | `share_encoder_denoiser=true`, `num_latent_tokens=8` |
| `us_unite_register_shared_nt16_s42` | shared | 16 | `share_encoder_denoiser=true`, `num_latent_tokens=16` |
| `us_unite_register_separate_nt4_s42` | separate | 4 | `share_encoder_denoiser=false`, `num_latent_tokens=4` |
| `us_unite_register_separate_nt8_s42` | separate | 8 | `share_encoder_denoiser=false`, `num_latent_tokens=8` |
| `us_unite_register_separate_nt16_s42` | separate | 16 | `share_encoder_denoiser=false`, `num_latent_tokens=16` |

Each row lives at
`egomimic/hydra_configs/model/bf/<row-id>.yaml`. These are the only row model
configs in the intended active set. Their `nt` spelling is the implementation
field name; in all scientific and collaborator-facing descriptions it means
latent-register count, never observation-token or patch-token count.

The manifest is the sole row authority. No patch-token arm, observation-token
count arm, or old parameter-count table belongs to this active set.

## Gradient telemetry contract and blocker

All six rows preserve the intended joint-update objective: every optimizer step
uses the reconstruction loss together with the aggregate flow loss. Their
wrapper controls are therefore:

```yaml
unite_flow_updates_per_reconstruction: 0
unite_gradient_telemetry_every_n_steps: 0
```

The telemetry cadence is disabled deliberately. In the current
`ModelWrapper.training_step`, the only call to shared-gradient telemetry is
nested under `unite_flow_updates_per_reconstruction > 0`. Consequently, a row
using the required joint-update setting cannot emit
`log/unite_gradient_cosine`, `log/unite_recon_grad_norm`, or
`log/unite_denoise_grad_norm`. The separate-topology metrics declared by the
manifest, `log/unite_tokenizer_recon_grad_norm` and
`log/unite_denoiser_flow_grad_norm`, do not yet have an emitter. The generic
`enable_grad_norm` control does not satisfy either topology contract.

Do not set `unite_flow_updates_per_reconstruction` above zero as a telemetry
workaround. That branch replaces the joint loss with alternating flow-only and
reconstruction-only optimizer steps, changing the training experiment.

Before launch, implement topology-aware gradient measurement that runs from the
same joint forward graph without changing the optimized loss. Then update the
smoke verifier, which currently requires alternating-update mode, and prove on
a real optimizer step that shared metrics are finite/nonzero and that separate
metrics are finite/nonzero while gradient cosine remains not applicable for the
separate topology. Until then, the metric names in the manifest are
requirements for a future launch gate, not claims about current runtime output.

## Corrected artifacts

- Sweep manifest:
  `unite_usocket_register_sweep_manifest.yaml`
- Interactive graph:
  `docs/config_graphs/unite_register_sweep_20260902/us-unite-register-sweep.html`
  - SHA-256: `5f4f42a9bc5f8ae48e23ab280064e2c78edaa8783c915e9d6540d7c4be4d45ad`
- Linted graph JSON:
  `docs/config_graphs/unite_register_sweep_20260902/us-unite-register-sweep.json`
  - SHA-256: `47d2b8747d62a6df680624a25ac5ecaa5cff58ca8fef7e4cfe585b855178a1ff`

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
- The external `flow_transfer_unite_skynet_x2_v20.sbatch` launcher is stale for
  this schema: it expects schema version 1 and emits the collaborator-facing tag
  `obs33tok`. It was intentionally not edited here and must not be used for this
  WIP sweep.

## Why launch remains blocked

1. All six numeric model/backbone parameter counts were invalidated. Each row
   needs a fresh construction-derived parameter manifest and SHA-256.
2. The exact episode split and train-only normalization artifacts remain
   unresolved.
3. The canonical launcher must be updated for schema 2 without `obs33tok`,
   patch-token, or exact-mechanism claims.
4. Joint-update-safe shared gradient telemetry is not implemented, the
   separate-topology gradient-norm metrics have no emitter, and the current
   smoke verifier assumes alternating-update mode.
5. No row has passed the required real optimizer-plus-validation smoke,
   checkpoint/EMA strict reload, required metric checks, or W&B visibility gate.
6. This collaborator branch is a reviewed implementation, not a clean immutable
   training run bundle.

Changing the manifest status by hand is not sufficient. It must remain blocked
until all six rows pass the canonical training gate and the exact corrected
artifacts are bound into an immutable bundle.

## Access and actions in this correction

`sky1` authenticated but did not execute a bounded trivial command, so the
Skynet access procedure required the `sky2` fallback. The current combined
review recorded:

- Python syntax/import checks, Ruff, and YAML parsing.
- `31` related tests across the released-policy, training-entry, and
  normalization-path suites. These include real H16 clean-action
  tokenize/denoise/backward coverage for shared/separate x `N={4,8,16}`,
  configured-wrapper selection, Muon/AdamW grouping, optimizer stepping, and
  optimizer-state round-trip.
- Canonical config-graph lint for all six rows in train and rollout modes
  (`12/12` graphs clean), recording `max_content_tokens=16`,
  `max_condition_tokens=1`, and enabled GE gradient checkpointing.
- The canonical trainer now honors `model._target_`; the released wrapper uses
  stable model-local parameter names; and the Torch 2.7.1 runtime uses a
  provenance-pinned backport of the official PyTorch Muon implementation.

This follow-up telemetry audit inspected the current wrapper and verifier source,
confirmed that no separate-topology telemetry emitter exists, parsed the
corrected manifest and all six row YAMLs, and checked the six row contracts for
exact agreement. It did not rerun the prior focused test suite or start a smoke.

No training job or W&B run was started.
