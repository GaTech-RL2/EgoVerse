# Articulated T demonstrations

This collector replaces the unusable articulated portion described in
`README_DATA_GENERATION.md`. The September 2026 run targets **162,000 accepted
episodes: 3,000 for each of nine embodiments in each of six control modes**.

- Embodiments: `u_socket`, `gripper`, `chain_gripper`, `suction`, `umi`,
  `triangle`, `scoop`, `flipper`, `spring`.
- Modes: `ideal`, `tight`, `loose`, `laggy`, `sticky`, `jittery`.
- Object: T; obstacle level: 0; images: 96 × 96; simulation rate: 30 Hz.
- New destination: `s3://rldb/staged/pushshapes_articulated/articulated-20260909/`.
- Existing datasets and the simulator implementation at `f952ca0d` are retained.

## What counts as an accepted demonstration

Every candidate must move the object at least 30 world units while interacting,
have at least 20 interaction frames, command orientation, and finish at coverage
≥ 0.95 while interacting. Command jerk/speed must be below 0.5. Position, angular,
and grip commands are rate limited, including the initial command.

The collector then replays **every action** in the original simulator. It saves
the episode only when maximum state error is below `1e-5`, object arena overflow
is below `1e-4`, and pusher/object penetration stays below `0.6`. Each episode is
written to a temporary Zarr directory and atomically renamed after completion.
Failed candidates and incomplete writes never count toward a quota.

The mechanisms differ:

| Embodiment | Required physical interaction |
| --- | --- |
| u_socket | Actual socket latch constraints |
| gripper, chain_gripper, suction, umi | Actual attachment constraints after aiming and actuating grip |
| triangle | Useful physical contact with changes in contact orientation |
| scoop | Physical arc contact and T material inside the concavity |
| flipper | Object movement during measured hinge motion |
| spring | Object movement while the plunger is compressed and stiffened |

The last four tools have no attachment constraint in this simulator. Their
quality check measures contact and mechanical work instead. The current audit
requires at least five useful work frames. Triangle and scoop additionally
require more than 0.1 radians of contact orientation travel.

The socket must reach the rear crossbar to latch: the working insertion depth
is about **−11**, rather than the positive depth suggested by the WIP note.
Gripper openings face local +Y, the socket faces +X, and the suction pad faces
−Y. Reachable T bar ends can be used when the stem points into a wall.

## Run locally

Activate the project environment before running Python:

```bash
source emimic/bin/activate
python -m Tsimulation.sim_v2.collect.articulation_probe \
  --embodiments gripper --seeds 10 --max-steps 1200 --output /tmp/gripper_probe
python -m Tsimulation.sim_v2.collect.articulation_collect \
  --embodiment gripper --gap ideal --target 10 --seed0 1000000 \
  --max-steps 1200 --out /tmp/gripper_demos
```

Use 2,500 steps for suction, 4,500 for triangle/flipper, and 5,000 for
scoop/spring. Difficult candidates are rejected rather than relaxed into the
accepted set. `ARTICULATED_FAST_SEARCH=1` enables tested broad-phase query
shortcuts during contact-tool search. Acceptance replay always uses the original
simulator. Search shortcut use is recorded in episode quality metadata.

Bulk collection uses 24 independent shards of 125 accepted episodes per cell.
Reset seed ranges are disjoint across embodiments, modes, and shards. For a
distributed run, `--shard-start` and `--shard-stop` select half-open shard
intervals; the denominator remains 24, so each shard still targets 125.

## Check completion and download

Use the existing R2 key-pair environment or the configured AWS Secrets Manager
credential. These tools do not print credentials. If using `~/.egoverse_env`,
export its variables in the current shell before running the commands.

```bash
source emimic/bin/activate
set -a
source ~/.egoverse_env
set +a
python -m Tsimulation.sim_v2.collect.articulation_status --output /tmp/articulated_status.json
python -m Tsimulation.sim_v2.collect.articulation_download \
  --snapshot /tmp/articulated_status.json --out /data/articulated \
  --embodiments u_socket --gaps ideal --shards-per-cell 1
```

The example downloads 125 socket episodes. Omit the embodiment, gap, and shard
filters to download all available shards. The status tool reports all 54 cells
and marks the run complete only after all quotas and nine batch completion
manifests are present. It checks archive sizes, checksum metadata, and disjoint
seed ranges. The downloader additionally hashes the complete downloaded bytes,
checks episode counts, and refuses to overwrite a different local shard.

Archives contain ordinary EgoVerse Zarr episodes under
`<gap>/<embodiment>/shardNNN/`. The downloader also creates unique episode
symlinks in `cells/<gap>/<embodiment>/`, the flat layout accepted by
`LocalEpisodeResolver`. Nothing is automatically registered for training.

```bash
python -m Tsimulation.sim_v2.collect.articulation_audit \
  --root /data/articulated --output /tmp/articulated_audit.json
```

This verifies every downloaded episode's numeric payload, action hash, JPEG
decoding, and quality bounds, then replays a sample from each downloaded cell.
`articulation_manifest_audit` checks the search-quality records and reset-seed
uniqueness for every uploaded episode using small archive range reads. Its
`--snapshot`, `--cache`, and `--output` arguments let later audits reuse already
verified manifests while generation continues.

## Schema and provenance

The main fields are `actions`, `observations.state`,
`observations.images.front_img_1`, `reward`, and `goal_pose`. Actions follow
the recorded `action_spec`: `(x, y, angle)` or `(x, y, angle, grip)`. State is
`(agent_x, agent_y, agent_angle, object_x, object_y, object_angle)` **before**
the action. `reward`, `engaged`, and `mechanism_work` refer to **after** the action.
`observations.mechanics` records jaw gap, UMI open fraction, chain joint angle,
flipper swing, and spring compression; its column names are in `mechanics_spec`.

Attributes include the reset seed, control-gap parameters, episode initial
state, quality measurements, action SHA-256, and collector version. The first
four grasping batches predate the extra `mechanism_work` column; they still
require measured attachment, transport, and full replay validation.

Each uploaded shard has a `.tar.json` completion record and SHA-256 metadata.
Frozen source capsules accompany batches or individual partition runs.
Checkpointed shards additionally record the preserved episode names, seeds,
action hashes, and source capsules in `checkpoint_source.json`; new episodes
record their source capsule SHA-256 directly.

The initial upload helper required a credential correction. Its affected OSMO
workflows can report failure after recovery has successfully persisted all
episodes. Use verified R2 completion manifests as the authority for data
completion. Slow contact jobs were subsequently checkpointed and restored into
disjoint partitions; their original job IDs are retired after restoration.

## Validation and previews

`tests/test_articulation_collection.py` checks physical success, exact replay,
Zarr/image round trips, overwrite protection, crash-resume accounting, and
optimized scoop search equality against the original simulator.

`articulation_preview` renders stored actions into H.264 videos, an HTML gallery,
and a contact sheet. The September run's local artifacts are under
`sim_run/articulated_demos_20260909/` beside the isolated worktree. The videos
play at 4× simulation speed and show measured interaction and goal coverage.
