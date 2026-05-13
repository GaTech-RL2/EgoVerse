# Tsimulation/

A small, hackable PushT-style sim plus mouse-driven data collection that
writes demos directly into EgoVerse's per-episode zarr format. Adapted
from [gym-pusht](https://github.com/huggingface/gym-pusht) and extended
with:

- pushable shapes (`T`, `U`, `Z`)
- pusher tools (`circle`, `stick`)
- obstacle levels (`0`–`3`)

## Install

The simulation deps are `pymunk`, `pygame`, `shapely` — already added to
[`requirements.txt`](../requirements.txt) and
[`pyproject.toml`](../pyproject.toml). `zarr`, `numcodecs`,
`opencv-python`, `simplejpeg`, `numpy` are already present in EgoVerse.

If your environment doesn't have them yet:

```bash
uv pip install pymunk pygame shapely
```

For headless boxes (no display) export `SDL_VIDEODRIVER=dummy` before any
`pygame.init()` — the tests do this automatically.

## Quick start

```bash
# 100 random steps; print final coverage.
python -m Tsimulation.examples.play_random --object T --pusher circle --obstacles 0
```

## Collect mouse demos

```bash
python -m Tsimulation.collect.mouse_collect \
    --output data/pushshapes_demos \
    --object T --pusher circle --obstacles 0 \
    --num-episodes 50 --image-size 96
```

Window hotkeys: **SPACE** record/pause, **S** commit + reset, **R** abort
+ reset, **Q** quit. Each committed episode becomes
`data/pushshapes_demos/episode_NNNNNN.zarr/`.

## Output schema

Per the conventions of
[`egomimic.rldb.zarr.ZarrDataset`](../egomimic/rldb/zarr/zarr_dataset_multi.py),
each episode is its own Zarr v3 store with:

| key                                 | shape    | dtype | notes                              |
| ----------------------------------- | -------- | ----- | ---------------------------------- |
| `observations.images.front_img_1`   | (T,)     | jpeg  | JPEG-encoded `(H, W, 3)` frames    |
| `observations.state`                | (T, 5)   | f32   | `[agent_x, agent_y, obj_x, obj_y, obj_theta]` |
| `actions`                           | (T, 2)   | f32   | target XY in 512-px world coords   |
| `reward`                            | (T, 1)   | f32   | per-step coverage IoU              |
| `goal_pose`                         | (T, 3)   | f32   | constant per episode               |

Metadata in `store.attrs`: `embodiment="pushshapes_sim"`,
`task_name="pushshapes"`, `task_description` (JSON of env args),
`total_frames`, `fps`, `features`.

Full schema rationale is in [`SCHEMA_NOTES.md`](SCHEMA_NOTES.md).

## Plugging into EgoVerse training

The output is consumable by the existing
`egomimic.rldb.zarr.zarr_dataset_multi.ZarrDataset` — point a key_map at
`observations.images.front_img_1`, `observations.state`, and `actions`.
The new embodiment `PUSHSHAPES_SIM = 15` is registered in
[`egomimic/rldb/embodiment/embodiment.py`](../egomimic/rldb/embodiment/embodiment.py).

## Tests

```bash
SDL_VIDEODRIVER=dummy pytest Tsimulation/tests/test_smoke.py -q
```

Covers all `{T,U,Z} × {circle,stick} × {0,1,2,3}` env smoke cases plus a
writer round-trip and resumability check.

## Layout

```
Tsimulation/
├── README.md
├── SCHEMA_NOTES.md
├── pushshapes/
│   ├── __init__.py    # registers PushShapes-v0
│   ├── env.py
│   ├── shapes.py
│   ├── obstacles.py
│   └── render.py
├── collect/
│   ├── __init__.py
│   ├── mouse_collect.py
│   └── zarr_writer.py
├── examples/
│   ├── play_random.py
│   └── replay_zarr.py
└── tests/
    └── test_smoke.py
```
