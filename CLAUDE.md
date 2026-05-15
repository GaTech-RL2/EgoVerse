# CLAUDE.md

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

### What is intentionally not implemented

- Annotation-level packed dataset (`ZarrAnnotationPackedDataset`). The
  `_read_span` primitive and `pack_collate` are written generically so
  this can be added as a sibling class with different index-building
  logic only.
- Hydra config knobs / wiring for the packed datasets.
- Model-side `cu_seqlens` plumbing and pos-emb adjustments.

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
