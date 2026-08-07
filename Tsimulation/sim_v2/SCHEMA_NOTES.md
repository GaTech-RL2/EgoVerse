# Schema Notes — what EgoVerse's zarr loader actually expects

Findings from auditing the host repo before writing the PushShapes collection
code. The original task prompt suggested a diffusion-policy / lerobot
single-store layout (`data/img`, `data/state`, `data/action`,
`meta/episode_ends`) — **EgoVerse does not use that layout**. The actual
schema is documented here so the writer in
[collect/zarr_writer.py](collect/zarr_writer.py) matches what the existing
reader at
[egomimic/rldb/zarr/zarr_dataset_multi.py](../egomimic/rldb/zarr/zarr_dataset_multi.py)
will load.

## Top-level layout: one .zarr per episode

A dataset is a directory of self-contained per-episode Zarr v3 stores:

```
<dataset_root>/
├── episode_000000.zarr/
├── episode_000001.zarr/
└── ...
```

Source: docstring of `ZarrDataset` in `zarr_dataset_multi.py`:

> Directory structure (per-episode metadata):
>     dataset_root/
>     └── episode_{ep_idx}.zarr/
>         ├── observations.images.{cam}  (JPEG compressed)
>         ├── observations.state
>         ├── actions_joints
>         └── ...
>
> Each episode is self-contained with its own metadata, enabling:
> - Independent episode uploads to S3
> - Parallel processing without global coordination
> - Easy episode-level data management

There is **no** global `meta/episode_ends` array — episode boundaries are
implicit in the directory tree. The lerobot-style replay-buffer layout from
the prompt does not apply.

## Inside one episode store

Each `episode_{idx}.zarr/` is opened as a Zarr v3 group. Two kinds of arrays:

1. **Numeric arrays** — one per feature, shape `(T, ...)`. Examples used in
   the repo: `observations.state`, `actions`, `actions_joints`,
   `actions_cartesian`. Bulk-written episodes are stored as one chunk per
   array so new collections match the compact rechunked layout.
2. **Image arrays** — one per camera, shape `(T,)` of
   `zarr.core.dtype.VariableLengthBytes`. Each element is a JPEG-encoded
   frame produced by `simplejpeg.encode_jpeg(img, quality=85, colorspace="RGB")`.
   Bulk-written episodes are stored as one chunk for the whole image array.
   Image keys are prefixed `observations.images.{cam}` by convention.

## Required metadata (store.attrs)

The reader does `self.metadata = dict(self._store.attrs); self.keys =
self.metadata["features"]`, so every store must populate:

| key                | type            | purpose                                                     |
| ------------------ | --------------- | ----------------------------------------------------------- |
| `embodiment`       | str             | Looked up via `get_embodiment_id` against the enum.         |
| `total_frames`     | int             | Episode length (used for length checks).                    |
| `fps`              | int             | Playback rate.                                              |
| `task_name`        | str             | Free-form.                                                  |
| `task_description` | str             | Free-form.                                                  |
| `features`         | dict[str, dict] | Per-key `{dtype, shape, names}`; `dtype="jpeg"` for images. |

## Image format

- Encoding: JPEG via `simplejpeg.encode_jpeg(..., quality=85, colorspace="RGB")`.
- Storage: `zarr.core.dtype.VariableLengthBytes()` dtype, one chunk per
  episode for bulk-written collections.
- Shape recorded in `features[key]["shape"]` as `[H, W, 3]`, dimension names
  `["height", "width", "channel"]`, `dtype` string `"jpeg"`.
- Most EgoVerse data is high-resolution (e.g. 480x640); for PushShapes we use
  96x96 to stay consistent with the gym-pusht expectation called out in the
  task prompt. The loader does not care about the exact resolution — it
  reads whatever was written.

## Compression

EgoVerse uses Zarr v3 plus JPEG for images. The compact bulk-write path now
matches the layout produced by `scripts/rechunk_zarr_dataset.py`, which keeps
per-episode file counts low while preserving the same data values.

## Embodiment

Embodiments are enumerated in
[egomimic/rldb/embodiment/embodiment.py](../egomimic/rldb/embodiment/embodiment.py).
To make `get_embodiment_id("pushshapes_sim")` resolve, we added
`PUSHSHAPES_SIM = 15` to the `EMBODIMENT` enum. The transform list is left
unset — the canonical model-side transforms in
`egomimic.rldb.embodiment.*` are robotic-arm-specific; PushShapes demos are
intended to be consumed by a separate code path that reads the raw zarr
state/action arrays.

## Writer plan for PushShapes

We use the existing `egomimic.rldb.zarr.ZarrWriter` rather than reimplementing
the format. `ZarrDemoWriter` in
[collect/zarr_writer.py](collect/zarr_writer.py) is a thin wrapper that:

- Targets `<output>/episode_{N:06d}.zarr` for each commit, picking the next
  index based on existing directories (resumability).
- Buffers a single episode in memory and calls
  `ZarrWriter.create_and_write(...)` on commit (atomic).
- `abort_episode()` simply discards the in-memory buffer — the on-disk store
  was never touched, so partial writes cannot corrupt anything.

### Per-frame keys we record

| key                                  | shape    | dtype      | meaning                                                |
| ------------------------------------ | -------- | ---------- | ------------------------------------------------------ |
| `observations.images.front_img_1`    | (H, W, 3) | uint8 → JPEG | Top-down render at `image_size`. Key matches `Embodiment.VIZ_IMAGE_KEY` so existing viz tools work. |
| `observations.state`                 | (5,)     | float32    | `[agent_x, agent_y, obj_x, obj_y, obj_theta]`.         |
| `actions`                            | (2,)     | float32    | Target XY in 512-px world coords (gym-pusht convention). |
| `reward`                             | (1,)     | float32    | Per-step coverage IoU.                                 |
| `goal_pose`                          | (3,)     | float32    | `[x, y, theta]` (constant per episode but stored per-frame for downstream convenience). |

### Metadata we set

- `embodiment="pushshapes_sim"`
- `task_name="pushshapes"`
- `task_description=json.dumps({object_shape, pusher_shape, obstacle_level, image_size, version})`

That lets the env that produced an episode be reconstructed exactly without
adding non-standard top-level attrs.

## Things we deliberately do NOT do

- No `meta/episode_ends` array — the loader doesn't read one.
- No `numcodecs.Blosc` configuration — Zarr v3 picks the codec via
  `ZarrWriter`.
- No raw `(N, H, W, 3) uint8` image storage — would not decode through the
  existing reader, which expects JPEG bytes.
- No new dataset loader — we plug straight into the existing
  `ZarrDataset` / `MultiDataset` reader path.
