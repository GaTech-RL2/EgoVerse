# CLAUDE.md

## SLURM / compute-node workflow

**Never run Python / training / eval / smoke scripts on the login node
(`sky1` — no GPU).** Always dispatch through a compute-node allocation:

```bash
# One-time: allocate an idle interactive node (keep several around so
# scripts can dispatch instantly without waiting on the queue).
salloc --no-shell <gpu-partition-args>     # e.g. via the geta40_rl2 alias

# Then run any script through the allocated job's id:
srun --jobid=<JOBID> bash -c "PYTHONPATH=. .venv/bin/python scripts/foo.py ..."
```

Use `squeue -u $USER -o '%i %j %T %N'` to list current allocations and
pick a `RUNNING` job to dispatch onto. Hostname-check inside scripts —
if a job thinks it's on `sky1`, abort.

## Packed dataloading (episode-level)

A variable-length, per-frame "packed" read path was added on top of the
existing per-frame `ZarrDataset`. The intent is full-episode loading: each
sample is one episode (or one chunk of a long episode), and a batch of
samples is concatenated into a single flat stream with `cu_seqlens`
boundaries (FlashAttention-style). The base read primitive and the
collator are deliberately generic so an annotation-level packed dataset
can be added later without rework.

### Files touched

- `egomimic/rldb/zarr/zarr_dataset_multi.py`
  - `ZarrDataset._annotations_for_span(start, end)` — collects every
    annotation whose `[start_idx, end_idx)` overlaps `[start, end)`.
  - `ZarrDataset._read_span(start, end, *, episode_idx=None)` — reads each
    `key_map` key over `[start, end)` with no horizon windowing and no
    padding. Handles JPEG/JSON decode and transforms exactly like
    `__getitem__`. Returns per-frame tensors of shape `(end - start, ...)`,
    plus `seq_len`, `embodiment`, `metadata.robot_name`, the configured
    annotation key as `list[str]`, and (optionally) `episode_idx`.
    `metadata_keys` are skipped; failures propagate so the wrapping packed
    dataset can resample.
  - `__getitem__` is unchanged.

- `egomimic/rldb/zarr/zarr_dataset_packed.py` (new)
  - `ZarrEpisodePackedDataset(datasets, max_seq_len, min_seq_len, chunking,
    max_resample_attempts)` — wraps a `dict[str, ZarrDataset]` (same shape
    returned by the existing resolvers).
    - Index is built once at construction: each entry is
      `(episode_key, start, end)`. Each `__getitem__` index = one chunk
      = one contiguous span from one episode.
    - `chunking="sequential"`: episodes longer than `max_seq_len` are
      split into consecutive non-overlapping chunks; the trailing chunk
      may be shorter (dropped if `< min_seq_len`).
    - `chunking="none"` (or `max_seq_len=None`): one entry per episode,
      raises if any episode exceeds `max_seq_len`.
    - On read failure (bad JPEG, transform error) it resamples to a
      random other index up to `max_resample_attempts` times.
    - Adds `chunk_offset` (start frame within the episode) to each sample
      on top of what `_read_span` returns.
    - Factories: `from_resolver(...)`, `from_local_folder(...)` — the
      latter is a convenience that builds a `LocalEpisodeResolver`
      internally for smoke tests.
  - `pack_collate(batch)` — generic collator:
    - Per-frame tensor keys (first dim equals the sample's `seq_len`):
      `torch.cat` along time → `(sum(seq_lens), ...)`.
    - List-valued keys (e.g. annotations): pass through as
      `list[list[str]]`, matching the existing `annotation_collate`
      contract.
    - Scalar-per-sample keys (`embodiment`, `episode_idx`,
      `chunk_offset`): collated into a `(B,)` tensor.
    - Emits `seq_lens` (LongTensor `(B,)`), `cu_seqlens` (LongTensor
      `(B+1,)`, `cu[0]==0`), `max_seq_len` (int), `batch_size` (int).

- `egomimic/pl_utils/pl_data_utils.py`
  - `MultiDataModuleWrapper` no longer hard-codes `annotation_collate`.
    A new helper `_collate_fn_for(dataset)` picks `pack_collate` when the
    dataset is a `ZarrEpisodePackedDataset`, otherwise `annotation_collate`.
    Each per-dataset DataLoader gets its own collate fn — a packed
    dataset can sit alongside unpacked datasets in the same combined
    loader.

### What is intentionally not implemented (still)

- Annotation-level packed dataset (`ZarrAnnotationPackedDataset`). The
  `_read_span` primitive and `pack_collate` are written generically so
  this can be added as a sibling class with different index-building
  logic only.

Wired up since this section was first written:

- Hydra config wiring for the packed dataset:
  `egomimic/hydra_configs/data/tsimulation.yaml` now targets
  `ZarrEpisodePackedDataset.from_resolver` with `chunking="none"`,
  `min_seq_len=64`, `batch_size=8`.
- Model-side `cu_seqlens` plumbing — see the "stage-based architecture +
  packed mode" and "algo-level packed training" sections below.
- Algo-side `HNetPolicy.forward_packed` (per-sub-sequence BOS shift +
  per-position `pos_emb` indexing + packed cond encoding).

### Smoke test

`scripts/smoke_packed_dataset.py` — runs against the pushT
`/coc/cedarp-dxu345-0/Tsim_datasets/test_demos` folder, builds a
`ZarrEpisodePackedDataset` (defaults `chunking="none"`), packs all
episodes through a `DataLoader(collate_fn=pack_collate)`, then writes:
- `packed_smoke_out/actions_packed.png` — every action dim plotted vs
  the packed frame index, with red verticals at each `cu_seqlens`
  boundary so per-episode segments are visible.
- `packed_smoke_out/ep{b}_{episode_name}_pack_t{first}_to_t{last}.mp4` —
  one full-playback MP4 per episode, capped at
  `N_EPISODES_TO_SAVE_VIDEOS` (default 2).

The script also cross-checks that the first/last `actions` frame at each
`cu_seqlens` boundary in the packed batch byte-matches a direct
per-frame read from the source `ZarrEpisode`.

### Test data

`/coc/cedarp-dxu345-0/Tsim_datasets/test_demos` — pushT, 4 episodes,
`embodiment="pushshapes_sim"`, action dim 2, image keys
`observations.images.front_img_1` (96×96 JPEG), state dim 5. No
annotations populated (so `annotations` lists come back empty).

`/coc/cedarp-dxu345-0/Tsim_datasets2` — newer pushT dataset, split by
shape: `circle/` (61 episodes, lengths 245–958 frames, median ~410) and
`stick/`. Each subfolder contains flat `.zarr` episode directories and is
what `LocalEpisodeResolver` should be pointed at directly (the resolver
does not recurse one level). The training data config
(`egomimic/hydra_configs/data/tsimulation.yaml`) and
`scripts/smoke_packed_dataset.py` currently target `circle/`.

## hnet_nets — kernel availability + fallback paths

The vendored H-Net stack at `egomimic/models/hnet_nets/` is written so the
optional CUDA kernels are detected at import time and silently swapped for
pure-PyTorch fallbacks when absent. Detection helpers live in:

- `blocks.has_flash_attn()` / `blocks.has_mamba()`
- `routing.has_mamba_scan()`

The cluster's system nvcc is `13.2` (under `/usr/local/cuda-13.2`); our
torch is `2.6.0+cu124`. Using the system nvcc directly would compile
against the wrong CUDA major version. The fix is to install a
**project-local cu12.4 toolkit** under `./cuda-12.4/` (see "Installing
the CUDA kernels" below) and point `CUDA_HOME` at it before building.

Fallback behaviour, in order of impact:

- `MultiHeadAttention._forward_packed` (`blocks.py:202`) uses SDPA with an
  explicit block-diagonal causal mask instead of `flash_attn_varlen_func`.
  Mathematically identical; slower and more memory-hungry for long packs.
- `DeChunkLayer._forward_padded` / `_forward_packed` (`routing.py:351, 378`)
  call `_ema_loop` (pure-Python `for t in range(M)`) instead of
  `mamba_chunk_scan_combined`. At init the chunker fires ~0% of the time
  so `M` is tiny and the loop is fast; once the ratio loss kicks in and
  `M` grows toward `T/8`, the Python loop becomes the dominant forward
  cost. **This is the main perf cliff to watch.**
- Mamba2 (`m` / `M` blocks) is unavailable; only `t` / `T` (transformer)
  arch tokens work in `arch_layout` strings. Current pushshapes config is
  pure-attention (`T4`), so this doesn't bite us yet.

## Installing the CUDA kernels (`flash_attn`, `mamba_ssm`, `causal_conv1d`)

We install a **project-local CUDA 12.4 toolkit** (matches `torch
2.6.0+cu124`) via `micromamba`, then build the three exts with the
`--no-deps --no-build-isolation` flags so pip cannot resolve and upgrade
torch out from under us.

### Why both flags matter

- `--no-build-isolation` only stops pip from spawning an isolated build
  env for the wheel — it does **not** stop pip from resolving and
  installing runtime deps at the top level.
- `--no-deps` is what actually prevents pip from touching torch. The
  ext wheels declare `torch` unpinned in their `Requires-Dist`, so a
  naked `pip install mamba_ssm` will happily upgrade torch to the
  latest (e.g. `2.12.0+cu130`), which then mismatches the cu12.4 nvcc
  we just installed and breaks `flash-attn`'s CUDA-version assertion.
- Always use **both** flags together when building CUDA exts in this
  venv.

### Step 1 — install a cu12.4 toolkit under `./cuda-12.4`

```bash
# 1a. Download micromamba (~18 MB) into the project.
mkdir -p .micromamba
curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C .micromamba bin/micromamba

# 1b. Install nvcc 12.4 + the matching headers / dev libs into ./cuda-12.4.
./.micromamba/bin/micromamba create -p ./cuda-12.4 \
  -c "nvidia/label/cuda-12.4.1" \
  cuda-nvcc cuda-cudart-dev cuda-cccl cuda-nvrtc-dev cuda-libraries-dev \
  -y

./cuda-12.4/bin/nvcc --version    # -> 12.4.131
```

The PyPI `nvidia-cuda-nvcc-cu12` wheel is **not enough** — it ships
`ptxas` + `nvvm` but **not** the `nvcc` driver binary. Use the conda
package via micromamba instead.

### Step 2 — build the three exts on a compute node

Login node `sky1` has no GPU and `torch.cuda.is_available()` is `False`,
which makes `setup.py` skip the CUDA build path. **Always run the build
through `srun --jobid=<JOB>`** against an interactive A40/A100 alloc.

The canonical build script lives at `scripts/build_cuda_exts.sh`. Run it
as:

```bash
srun --jobid=<JOB> --chdir=$PWD scripts/build_cuda_exts.sh
```

The script does, in order:

```bash
export CUDA_HOME=$PWD/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # A100, A40, L40, H100
export MAX_JOBS=4                              # cap nvcc -j to avoid OOM

# --no-deps so pip can't upgrade torch; --no-build-isolation so the
# build uses the project's torch instead of a fresh isolated env.
.venv/bin/python -m pip install --no-deps --no-build-isolation causal_conv1d
.venv/bin/python -m pip install --no-deps --no-build-isolation mamba_ssm
.venv/bin/python -m pip install --no-deps --no-build-isolation flash-attn
```

Total wall time on an A40: ~25-40 min (flash-attn is the long pole).

### Step 3 — verify

```bash
.venv/bin/python -c "
from egomimic.models.hnet_nets.blocks import has_flash_attn, has_mamba
from egomimic.models.hnet_nets.routing import has_mamba_scan
print('has_flash_attn :', has_flash_attn())
print('has_mamba      :', has_mamba())
print('has_mamba_scan :', has_mamba_scan())
"
# All three should be True. ``has_mamba_scan = True`` is the one that
# removes the EMA Python loop cliff documented above.
```

### Recovery — if pip upgraded torch by accident

If you forget `--no-deps` and the env ends up with `torch 2.12+cu130`,
the recovery is in `scripts/recover_torch_cu124.sh`:

```bash
.venv/bin/python -m pip uninstall -y causal_conv1d mamba_ssm
.venv/bin/python -m pip install --no-deps --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
# then re-run build_cuda_exts.sh with --no-deps.
```

## hnet_nets — stage-based architecture + packed mode

The H-Net is built as a flat list of stages that `HNet.__init__`
(`hnet_nets/hnet.py`) wires into a recursive chain via
`stages[i].inner_stage = stages[i+1]`. Three stage types exist:

- `EncoderDecoderStage` — Isotropic encoder + inner_stage + closure-residual
  + Isotropic decoder. Same hidden dim in/out.
- `ChunkerStage` — RoutingModule + ChunkLayer + (optional dim-bridge
  Linear) + inner_stage + (dim-bridge Linear) + DeChunkLayer + STE residual
  gate. Registers `bpred` into `ctx.aux` for the ratio loss.
- `ComputeStage` — terminal Isotropic main_network.

Each stage's `forward(x, ctx)` branches on `ctx.packed` (=
`ctx.cu_seqlens is not None`):

- **Padded mode** (default) — `x: (B, T, D)`, stages build a `mask =
  ones(B, T)` and pass to Isotropic / Routing / Chunk / DeChunk.
- **Packed mode** — `x: (T_total, D)`, stages pass `cu_seqlens` /
  `max_seqlen` straight through to sub-modules. `ChunkerStage` substitutes
  the **chunked-space** `next_cu_seqlens` / `next_max_seqlen` onto `ctx`
  for the call into `inner_stage` and restores the outer values
  afterwards (try/finally). The cond contract in packed mode is
  `ctx.cond_dict[cond_key]: (T_total, d_cond)`.

`step()` paths are autoregressive single-token inference and do **not**
use `cu_seqlens` — packed mode only matters during training.

Both of those have since been wired — see the algo + data-config sections
below.

## hnet_nets — pushshapes obs alignment

`egomimic/rldb/embodiment/pushshapes.get_keymap` now sets `horizon:
action_horizon` on `front_img_1` and `state_agent_obj` (not just on
`actions`). This switches the padded dataloader from single-frame
broadcast obs to per-frame `(B, T, ...)` obs, so AdaLN sees
`cond_encoder(obs_t)` at each timestep instead of
`cond_encoder(obs_0)` broadcast. `SimpleConv` already accepts `(B, T, C,
H, W)` via leading-dims flatten/unflatten; `CondEncoderModule.encode`
skips its `unsqueeze-and-expand` branch when `x.dim() != 2 / 4`.

In packed mode `ZarrDataset._read_span` ignores the keymap horizon and
returns true per-frame obs anyway, so the change keeps the padded and
packed paths consistent.

## A40 memory baseline (full episode, packed, no chunking)

`scripts/debug_full_episode_mem.py` — packed forward+backward on the
longest episode (T=958) in `Tsim_datasets2/circle/` through the full
pushshapes arch (5.71M params, fp32, B=1).

| router behaviour    | inner stage sees | peak fwd | peak bwd |
|---------------------|-----------------:|---------:|---------:|
| default (init, ~0% boundary fire) | ~1 token | 1.73 GiB | 1.75 GiB |
| forced all-boundary (worst case)  | 958 tokens | 1.83 GiB | 1.83 GiB |

A40 has 44.4 GiB. Plenty of headroom — easily fits B>>1 packed batches.

## hnet_nets — algo-level packed training

`egomimic/algo/hnet.py` is the Lightning-side algo. The legacy
``data_schematic`` constructor argument has been **removed** entirely — it
referenced a class that didn't exist in this repo and the algo's
``normalize_data`` / ``unnormalize_data`` calls used method names that
weren't on ``MultiDataset``. The replacement is the ``norm_stats``
parameter (a ``MultiDataset`` stats holder, the same one ``HPT`` uses) that
``pl_model._instantiate_model`` already passes via Hydra. Method calls
moved to ``self.norm_stats.normalize(...)`` /
``self.norm_stats.unnormalize(...)`` / ``self.norm_stats.zarr_key_to_keyname(...)``.

### Packed training step path

- `HNetPolicy.forward_packed(actions_packed, obs_packed, cu_seqlens, max_seqlen)`
  is the packed analog of ``forward``:
  1. ``action_in`` tokenization on the flat ``(T_total, action_dim)`` actions.
  2. Global shift-right by 1, then overwrite every ``cu_seqlens[:-1]`` slot
     with the (cast-to-activation-dtype) BOS token. So each sub-sequence
     sees BOS at its start and the prior actions at later positions —
     identical semantics to the padded path but per-sub-sequence.
  3. ``pos_emb`` indexed by ``local_pos = t - cu_seqlens[seq_idx[t]]`` so
     position 0 of every episode lines up with ``pos_emb[0]``. Asserts
     ``max_seqlen <= action_horizon``.
  4. Cond encoding via the existing ``CondEncoderModule.encode``, which
     accepts ``(B, T, ...)`` per-frame obs — we feed
     ``(1, T_total, ...)`` and squeeze to ``(T_total, d_cond)``.
  5. Set ``ctx.cu_seqlens / ctx.max_seqlen`` and call ``self.hnet``;
     ``ctx.packed`` is True so every stage routes through its packed branch.
- `HNet.process_batch_for_training` detects packed batches by the
  presence of ``"cu_seqlens"`` and forwards the ``_PACKED_META_KEYS``
  (``cu_seqlens``, ``max_seq_len``, ``seq_lens``, ``batch_size``,
  ``embodiment``, ``episode_idx``, ``chunk_offset``) through unchanged.
  Data tensor keys are key-name resolved via
  ``norm_stats.zarr_key_to_keyname``. Sets ``processed[emb_id]["_packed"]``
  for the downstream dispatch and sets ``pad_mask=None`` (no padding to
  mask in a packed stream). Normalization runs **after** the keyname
  resolve via ``self.norm_stats.normalize(processed[emb_id], emb_id)`` —
  this is shape-agnostic per-feature broadcasting, so it works equally on
  padded ``(B, T, D)`` and packed ``(T_total, D)``.
- `HNet.forward_training` dispatches on ``_packed``: packed batches go to
  ``HNetPolicy.forward_packed``; padded batches keep the existing call.
  Per-chunker logging stats (``avg_chunk_len_i`` and ``boundary_rate_i``
  plus aggregates) come from ``chunk_stats_from_aux`` and surface through
  ``compute_losses`` → ``log_info`` as ``emb{id}_avg_chunk_len`` etc.

### Validation path (packed + AR rollout)

- `HNet.forward_eval` dispatches: packed batches → ``_ar_rollout_packed``;
  padded batches keep the legacy ``policy.generate(B, T=action_horizon)``.
- `HNet._ar_rollout_packed` slices each sub-sequence's obs into
  ``(1, T_ep, ...)`` and calls ``HNetPolicy.generate(obs, batch_size=1,
  T=T_ep)`` for that episode's exact length. Predictions are packed into
  ``(B, T_max, action_dim)`` zero-padded so downstream metric/viz code
  treats variable-length episodes uniformly. The companion key
  ``emb{id}_seq_lens`` is emitted alongside so the evaluator can mask
  zero-padded positions out of MSE.
- `HNetPolicy.generate(obs, batch_size, device, T=None)` got an optional
  ``T`` parameter (defaults to ``action_horizon``) so per-episode rollout
  doesn't always walk all 1024 pos_emb slots.

### Eval class (`egomimic/eval/eval_hnet.py`) + config

- `HNetEvalVideo` is a stripped-down version of ``HPTEvalVideo`` without
  HPT-specific machinery (no ``shared_ac_key``, no ``auxiliary_ac_keys``,
  no reverse-KL, no transform-list cam-frame projection).
- Packed batches: unpacks GT actions to ``(B, T_max, D)`` padded, computes
  **masked MSE** over only valid positions (``arange(T_max)[None, :] <
  seq_lens[:, None]``); also reports ``final_mse`` at each episode's last
  valid index. Builds the viz batch by slicing each episode's frame-0
  image and feeding it to the configured ``viz_func``
  (``pushshapes.viz_gt_preds``).
- `egomimic/hydra_configs/evaluator/eval_hnet.yaml` pulls in
  ``viz/cartesian.yaml`` and caps ``limit_val_batches: 4`` because AR
  rollout at ``action_horizon=1024`` is slow.

### Bug fixes uncovered while wiring this up

- **`ZarrDataset.__getitem__` multi-frame JPEG decode (`zarr_dataset_multi.py`)**:
  pre-fix, ``simplejpeg.decode_jpeg`` was called once on an array of JPEG
  buffers when ``horizon > 1`` on an image key — crashed with
  ``"Buffer dtype mismatch, expected 'const unsigned char' but got Python
  object"``. The retry loop then iterated ``total_frames`` times before
  giving up (393 s hang per probe). Fixed by looping per-frame in the
  horizon branch (mirrors what ``_read_span`` already did).
- **`MultiDataset._iter_leaves`**: now descends into
  ``ZarrEpisodePackedDataset.datasets`` so ``populate_from_datasets`` sees
  the inner ``ZarrDataset`` leaves and registers their key types. Without
  this, packed datasets came out of populate with empty ``key_types``.
- **`MultiDataset.populate_from_datasets`**: probes each embodiment exactly
  once (skips duplicate leaves). All 61 packed-dataset leaves share the
  same ``key_map`` so probing each was wasteful.
- **`MultiDataset.infer_norm_from_dataset`**: detects
  ``ZarrEpisodePackedDataset`` and uses ``pack_collate``; without that,
  ``default_collate`` torch.stacks variable-length episode tensors and
  crashes. ``sample_frac`` is now interpreted as a **frame** budget in
  packed mode (computed off ``sum(end-start for ds.index)``) instead of an
  episode budget, so we don't terminate after 61 frames out of ~28k.
- **`ZarrEpisodePackedDataset.set_norm_stats_from`**: added as a no-op for
  ``trainHydra`` parity. ``MultiDataset.set_norm_stats_from`` wires stats
  into the dataset so ``__getitem__`` normalizes at read time; the packed
  dataset deliberately keeps reads raw and lets the algo normalize.
- **bf16 / fp32 mismatches under autocast**:
  - ``HNetPolicy.forward_packed`` casts ``bos`` and ``pos_emb[local_pos]``
    to ``a_emb.dtype`` so index-put and addition work under
    ``trainer.precision=bf16``.
  - ``DeChunkLayer.step`` casts ``boundary_prob[boundary_mask, -1].clamp(…)``
    to ``p.dtype`` (the inference-cache dtype, fp32) before the index-put.
- **`HNet.process_batch_for_training` typo**: the original ``if key is
  not None`` was always true; should have been ``if key_name is not
  None`` so unrelated batch keys (e.g. ``metadata.robot_name``) don't end
  up bucketed under the ``None`` key.

## hnet_nets — training recipe (init + LR + WD)

Ported from upstream ``hnet/models/hnet.py`` + ``hnet/utils/train.py``,
adapted to the flat stage list:

- `apply_optimization_params(param, **kwargs)` (in `hnet_nets/hnet.py`):
  stamps a parameter with an ``_optim`` dict (merges on repeat).
- `HNet.init_weights(initializer_range=0.02)`: walks the stage chain
  applying residual-stream-aware Linear init. ``out_proj`` (attention)
  and ``fc2`` (MLP) weights get
  ``std = initializer_range / sqrt(n_residuals)``; other Linears get
  plain ``initializer_range``. Modules marked ``_no_reinit`` (routing
  q/k = identity, ``residual_proj`` = zero, AdaLN proj = zero) are
  skipped.
- Per-stage hooks: ``EncoderDecoderStage._init_weights`` adds
  ``encoder.height + decoder.height``; ``ComputeStage._init_weights``
  adds ``main_network.height``; ``ChunkerStage._init_weights`` adds 0
  (chunker doesn't contribute residual-stream depth). Recursion threads
  ``n_residuals`` through ``inner_stage``.
- `HNet.apply_lr_multiplier(list[float])`: stamps every parameter in
  stage ``i`` with ``lr_multiplier=multipliers[i]``. Wrong list length
  raises ``IndexError`` immediately. Recurses via ``inner_stage`` with
  ``stage_idx+1``.
- `HNet.parameter_groups(weight_decay=0.0)`: builds
  ``list[dict]`` for ``AdamW(params=...)``. Bias and norm-weight params
  always get ``weight_decay=0`` (name-based detection: ``*.bias``,
  ``*.norm*``, ``*rmsnorm*``). All other params are grouped by their
  ``_optim`` tuple. Returns groups with ``params``, ``weight_decay``, and
  ``lr_multiplier``; the caller is expected to multiply ``lr_multiplier``
  into ``lr`` before passing to AdamW.

**Status**: methods are on ``HNet`` and pass 20 unit tests. They are
**opt-in** — neither ``HNetPolicy.__init__`` nor
``pl_model.configure_optimizers`` call them automatically yet. To use:

```python
hnet.init_weights(0.02)
hnet.apply_lr_multiplier([3.0, 1.7, 0.9])   # outer → inner
groups = hnet.parameter_groups(weight_decay=0.01)
for g in groups:
    g["lr"] = base_lr * g["lr_multiplier"]
optimizer = torch.optim.AdamW(groups)
```

Works for any flat stage composition (1-stage, 2-stage, multi-chunker, …),
because both methods recurse through ``inner_stage`` and use position-in-list
as the stage index.

## trainHydra invocation

Working invocation for the pushshapes packed run (debug-sized — 4 epochs,
2 train batches each, 3 val batches every 2 epochs):

```
python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_smoke description=trainhydra_test mode=train \
  data=tsimulation model=hnet_pushshapes evaluator=eval_hnet \
  trainer=debug logger=debug '~callbacks'
```

Notes:

- `logger=debug` resolves to ``egomimic/hydra_configs/logger/debug.yaml``
  which is intentionally empty (no logger, no wandb). Older
  ``logger=null`` syntax doesn't work — Hydra rejects ``=null`` for
  config-group overrides.
- `~callbacks` removes the callback group (avoids the default
  ``checkpoints`` callback). Same idiom for any group not needed.
- The norm-stats step iterates ~28k frames at ~500 sample/s ≈ 1 min on
  first run; subsequent runs can use
  ``norm_stats.precomputed_norm_path=/coc/.../norm_stats`` to skip it.
- ``trainer=debug`` extends ``trainer/ddp.yaml`` and sets:
  ``limit_train_batches=2``, ``limit_val_batches=3``,
  ``check_val_every_n_epoch=2``, ``max_epochs=4``, ``profiler=simple``.
- ``action_horizon`` is hard-coded as ``1024`` in **both** the data
  YAML (``tsimulation.yaml`` ``get_keymap`` call) **and** the model
  YAML (``hnet_pushshapes.yaml``). Hydra's
  ``hydra.utils.instantiate(cfg.data, ...)`` passes every top-level key
  to ``MultiDataModuleWrapper.__init__``, which doesn't accept
  ``action_horizon``, so a top-level interpolation source isn't an
  option. They must match.

## Smoke scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `smoke_packed_dataset.py` | Verifies `ZarrEpisodePackedDataset` byte-matches direct per-frame zarr reads; writes per-episode MP4s and an actions plot with cu_seqlens verticals. |
| `smoke_packed_norm_stats.py` | End-to-end norm-stats collection on the packed dataset (catches `_iter_leaves`, pack_collate, populate_from_datasets bugs). |
| `smoke_packed_training.py` | Direct forward+backward through `HNet` (the stage tree) with a packed batch — bypasses the algo wrapper, exercises stage-level packed plumbing. |
| `smoke_packed_training_e2e.py` | Goes through the **algo** path: `process_batch_for_training` → `forward_training` → `compute_losses` → backward. Includes normalize. |
| `smoke_packed_validation.py` | Per-episode AR rollout through `HNetEvalVideo.compute_metrics_and_viz`; produces metrics + a viz frame per episode. |
| `debug_full_episode_mem.py` | A40 memory probe: forward+backward on one full episode (default chunker AND forced-all-boundary worst case). Prints peak GiB. |

## Tests (`tests/`)

| File | Coverage |
|---|---|
| `test_hnet_nets.py` (57 tests) | RoutingModule / ChunkLayer / DeChunkLayer / Isotropic / stages (padded + packed) / HNet assembly / ratio_loss / chunk_stats / STE / RMSNorm / AdaLN. |
| `test_packed_pipeline.py` (9 tests) | normalize broadcast on padded vs packed; `_iter_leaves` descent into packed; `__getitem__` multi-frame JPEG decode (real episode); end-to-end packed stats collection. |
| `test_training_recipe.py` (20 tests) | `apply_optimization_params`, `init_weights` height-scaled init (out_proj/fc2 vs other Linears, _no_reinit modules untouched), `apply_lr_multiplier` per-stage stamping, `parameter_groups` (default, with bias/norm WD=0, per-stage groups, AdamW-consumable). Plus flexible-config combinations: single ComputeStage, ComputeStage→ComputeStage, EncoderDecoderStage→ComputeStage. |

Full suite: `python -m pytest tests/ -q` (86 tests, ~20s).
