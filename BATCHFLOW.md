# BATCHFLOW — the one-page convention (2026-07-10)

**Every module is `stage(batch: dict) -> dict`.** One carrier, one interface.
Stages read keys, compute, write keys. Nothing else exists: no `HNetContext`,
no `StreamBundle`, no attribute stashes, no forward hooks for probes, no side
channels. The model is an ordered list of stages built from the yaml.

## The batch dict

| namespace | meaning | producer |
|---|---|---|
| `actions`, `__obs`, `cu_seqlens`, `max_seq_len`, `seq_lens` | data + packing meta | dataloader (pack_collate) |
| `embodiment` | the ONE embodiment identifier (string) | runner per-emb loop |
| `frame_idx` | original within-episode frame index per token | TargetBuilder |
| `x` / `A`, `S` / `a_top`, `s` / `cvae_z` … | features, accumulated as computed | stages |
| `pred_action` | model output | head stage |
| `loss/*` | scalar loss terms; runner sums them | loss stages |
| `log/*` | scalar diagnostics; runner logs as `Train/{emb}/<name>` | any stage |
| `aux/*` | accumulator LISTS (ratio-loss entries, probe payloads) | stages append |

## Rules

1. **One interface.** If a change needs a new signature, the design is wrong —
   add keys instead.
2. **Contracts.** Every stage declares `reads` / `writes`. `Pipeline.plan()`
   validates at build time and resolves modes: rollout seeds the dict with
   obs-only → posterior/loss stages are excluded PROVABLY at plan time
   (reported once), never silently skipped.
3. **Recursion = `sub_batch`.** A stage recursing at another temporal
   resolution passes `sub_batch(batch, x=..., cu_seqlens=next_cu, ...)` — a
   shallow copy with overridden views. Per-level state stays in locals.
   Mutate-and-restore is banned (the old ctx save/swap/restore, and the
   diffusion outer stages' mutate-WITHOUT-restore bug, are both
   unrepresentable).
4. **Packed ops live in `egomimic/pipeline/packed.py` only.** No module
   reimplements cu_seqlens bookkeeping.
5. **Losses are stages.** They write `loss/<name>`; the runner sums `loss/*`.
   Validation gets val losses for free.
6. **No dead knobs.** Every config key is read by something; strict-consume
   enforced. (The enable_grad_norm disaster class.)
7. **Dtype policy at seed time.** All tensors normalized to the model dtype
   when the dict is seeded; rollout state IS a persistent batch dict, so the
   bf16-rollout-state bug class is unrepresentable.
8. **Delete, don't deprecate.** Net-negative line count per refactor.

## Config shape

Stages live in three modules by role — `stages_io` (I/O, encoders, heads,
losses), `stages_hnet` (the chunking/trunk hierarchy) and `stages_flow`
(flow-matching and diffusion heads). The runner is `pipeline.algo.PipelineAlgo`.

```yaml
model:
  robomimic_model:
    _target_: egomimic.pipeline.algo.PipelineAlgo
    stages:
      - _target_: egomimic.pipeline.stages_io.TargetBuilder      # writes: actions targets, frame_idx
      - _target_: egomimic.pipeline.stages_io.ObsEncoders        # writes: A, S
      - _target_: egomimic.pipeline.stages_hnet.DualstreamTrunk  # reads: A, S -> a_top, s
      - _target_: egomimic.pipeline.stages_hnet.DualTrunkLevel   # L0 A256/S64
      - _target_: egomimic.pipeline.stages_hnet.DualChunkerLevel # 256->512 / 64->128
      - _target_: egomimic.pipeline.stages_hnet.ApexLevel        # agnostic-only apex
      - _target_: egomimic.pipeline.stages_flow.SDPHead          # -> pred_action, loss/*
      - _target_: egomimic.pipeline.stages_io.RatioLoss          # -> loss/ratio, log/*
```

That is the shape of `model/bf_rh_sdp_dual.yaml`; see it for the full,
runnable stage list. Overrides address stages by index
(`model.robomimic_model.stages.3.z_dim=16`). Swapping heads = swapping list
entries.

A CVAE variant instead uses `stages_io.CVAEPosterior` (reads S, actions ->
cvae_z, loss/kl) with `stages_io.CVAEHead` as the decoding head.

## Lineage

Old code (pre-refactor) is FROZEN at `EgoVerse-gmm-dualstream` (sky) /
branch `elmo/dualstream-campaign-moe-dtw` — use it to run old checkpoints.
This lineage: repo `EgoVerse-batchflow`, branch `elmo/batchflow-core`+.
Backward ckpt compatibility explicitly NOT required (user 2026-07-10).
