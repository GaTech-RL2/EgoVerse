# Zarr → Lance migration plan

Plan for converting every non-deleted zarr episode in `app.episodes` (all in R2 under
`s3://rldb/processed_v3/`) into the Lance tables. Aria and EVA episodes are converted from
their raw captures only (§9). Parts:

- `egomimic/scripts/lance_pilot/plan_zarr_migration.py`: read-only planner. It writes the manifest, the rejected list and the inventory.
- `egomimic/scripts/zarr_to_lance_ray.py`: Ray driver for the zarr route. It reads the manifest and calls `zarr_to_lance.convert_zarr_episode`.
- `egomimic/scripts/lance_commit.py --source zarr`: the committer, which stays the only writer.

## 1. Inventory (SQL snapshot 2026-10-30, R2 sampled)

Planner run v2 (`/coc/flash7/scratch/lance_pilot/migration_plan/v2/`, SQL read
2026-10-30 with the yam and dexmate steps schemas and Aria/EVA on the raw route only):
**289,734 episodes (514M frames) are convertible** into 23 steps tables, in 4,539 work
units: 284,402 on the zarr route (499M frames) and 5,332 on the raw route (15.0M frames).
335,892 are rejected, all but 141 of them for a missing operator (microagi v1 and trace).
99 are already committed in the pilot root `lance_v2`. The rest of this section is the first
run (before the robot schemas), whose R2 byte estimates still hold.

The table has 633,920 rows. 625,626 are not deleted, with 722M frames and an estimated
**60 TB of zarr on R2**. The byte estimate comes from listing 40 random zarrs per
(lab, embodiment), 1,698 zarrs in all, and scaling each stratum's bytes/frame by its SQL
frame count. **None of the 1,698 sampled zarr prefixes was missing.** A full existence check
would need ~625k list calls. The driver detects a missing zarr anyway (the download fails
and the episode lands in the report).

| kind | lab | embodiment | end_effector | episodes | convertible | frames (M) | est. zarr TB | blocked by |
|---|---|---|---|---:|---:|---:|---:|---|
| human | microagi | human_bimanual | human_hand | 514,239 | 186,188 | 433.5 | 24.0 | 328,051 v1 rows have no operator |
| robot | abc | yam_bimanual | parallel_jaw_gripper | 36,882 | 36,882 (v2) | 146.0 | 26.6 | |
| human | mecka | human_bimanual | human_hand | 41,617 | 41,611 | 89.5 | 5.0 | missing task |
| human | scale | human_* | human_hand | 17,090 | 17,090 | 23.4 | 1.5 | |
| human | trace | human_bimanual | human_hand | 7,700 | 0 | 11.0 | 1.8 | no operator (all) |
| human | rl2 | human_* (aria) | human_hand | 1,609 | 1,604 | 6.2 | 0.3 | |
| robot | rl2 | eva_bimanual / eva_right_arm | parallel_jaw_gripper | 2,879 | 2,711 | 3.3 | 0.5 | 168 have no zarr (raw only) |
| robot | rl2 / yam | yam_bimanual | parallel_jaw_gripper | 2,519 | 2,516 (v2) | 3.2 | 0.5 | 3 rl2 rows have no zarr |
| human | eth, song, wang (aria) | human_* | human_hand | 971 | 966 | 5.7 | 0.3 | |
| robot | sharpa_dex | dexmate_bimanual | dex_hand | 115 | 115 (v2) | 0.2 | 0.03 | |
| — | rl2 | `eva`, `aria` (legacy names) | — | 5 | 0 | 0.01 | — | unknown embodiment |

First-run totals: 250,168 episodes (364M frames, ~20.7 TB of zarr) were convertible in 19
steps tables; the yam and dexmate schemas add 39,516 episodes (149M frames, ~27 TB). The
converter handles every zarr-route format: mecka, scale, microagi v2, yam and dexmate.

Rejections (`rejected.parquet`; a row can have several reasons):

| reason | rows | notes |
|---|---:|---|
| missing_operator | 335,751 | microagi v1 (`processed_v3/microagi/`, 328,051: also no scene, rig_name, license) and all of trace (7,700) |
| no_steps_schema | 39,516 | yam_bimanual and dexmate_bimanual; 0 in v2 |
| no_zarr_path | 181 | mostly rl2 eva (168); 128 in v2, where EVA rows with raw take the raw route |
| no_frames / zarr_processing_error | 62 / 55 | num_frames ≤ 0 ("Zero Frames") |
| missing_task | 8 | mecka (6); 2 more have a placeholder task such as "debug" |
| unknown_embodiment | 5 | rl2 `eva` (4), `aria` (1): legacy names |

Scene is blank on 8,433 convertible rows. Scene is waivable, so those rows go through.

**microagi v1 vs v2.** The two sets cover the same capture dates (2025-05 to 2026-07). v1
uses snake_case task names and v2 uses free text. No (task, num_frames, scene) tuple is
shared between them. v2 looks like a re-delivery of the same captures, but SQL cannot
show that. The owner needs to confirm whether v1 should be migrated at all (see Open
decisions).

## 2. Destination layout

A new root, `s3://rldb/lance/`, separate from the pilot roots. It follows the existing
committer layout:

```
s3://rldb/lance/
  recordings.lance  tasks.lance  aliases.lance  changes.lance  annotations.lance
  steps/{lab}/{embodiment}/{end_effector}.lance        # 23 tables for the convertible set
  videos/{recording_id}/{content_digest[:16]}/{cam}.mp4
s3://rldb/lance_staging/zarr/{vendor_recording_id}/    # steps.parquet, *.mp4, meta.json last
```

`recording_id = recording_id(source, vendor_recording_id)`, chosen as follows:

- **aria / eva** (raw route only) use `source` = `aria` or `eva` and the vendor id is the
  epoch ms that the episode hash encodes, as the raw converters compute it. The 99 pilot
  recordings in `lance_v2` have the same ids.
- **every other lab** uses `source = "zarr"` and the vendor id is the `episode_hash` itself.
  Mecka and abc hashes are not timestamps. The id is deliberately not keyed by lab, so
  correcting a lab in SQL does not change the id.

## 3. Sharding and work units

- **Shard** = one steps table, `{lab}/{embodiment}/{end_effector}`.
- **Work unit** = 64 episodes of a single shard and route, sorted by episode_hash (4,539
  units in v2). A unit never spans steps tables, so a committer batch built from staged units touches few
  tables.
- Only one committer runs per root, not one per steps table. `recordings`, `tasks`,
  `aliases` and `changes` are shared by every steps table, and every batch writes and tags
  all of them. Parallel committers would conflict on those tables. Sharding exists to make
  each batch's steps commits small, not to parallelise the commits.
- Rollout and retries select by shard glob (`--shard 'rl2/eva_*'`), by work unit, or by
  `--limit`.

## 4. Worker flow (`zarr_to_lance_ray.py`)

The driver submits at most `--max-in-flight` tasks at a time. Each Ray task (4 vCPU by
default) handles one episode:

1. List and download `zarr_uri` from R2 to local NVMe/EBS. Nothing runs from a shared filesystem.
2. Call `convert_zarr_episode(local_zarr, out, metadata=<manifest row>, episode_hash=…)`.
   The converter writes `steps.parquet`, AV1 MP4s and `meta.json`, and rejects placeholder
   values for lab, task_name and operator. It refuses Aria and EVA zarrs, microagi v1 and
   trace. The zarr's `annotations` go into `meta.json` (see Annotations below).
3. Check `meta.recording_id == manifest.recording_id`. The task fails on a mismatch, which
   catches any planner/converter drift. A difference between `meta.num_frames` and SQL
   `num_frames` does not fail the task; it is recorded in the report.
4. Stage to `lance_staging/zarr/{vendor id}/` with `commit.stage`, uploading meta.json last
   (`commit.stage_direct` with `--direct-videos`: MP4s go straight to the Lance root, see below).
5. Delete the local files and append one JSON line to `--report`.

The committer (`lance_commit --source zarr --batch-size N`) runs on a loop on one EC2
instance. Each pass picks up everything with a complete `meta.json`, validates it, commits
in batches (one `Append` per touched table, then a tag) and unstages what it committed.

### Gap versus the decided write model

The decided model (2026-10-15) is that workers call `write_fragments` straight into the
steps tables, and the committer does one `LanceOperation.Append` with the fragment metadata.
The committer still downloads each staged delivery's parquet and writes steps with
`lance.write_dataset`.

Moving the MP4s through the committer would mean ~3–5 TB passing through one machine twice.
Two options were considered:

- (a) workers upload MP4s straight to the content-addressed `videos/{rid}/{digest}/`, and
  `meta.json` carries the digest and paths. **Implemented** (`zarr_to_lance_ray
  --direct-videos`, `commit.stage_direct`): the worker uploads each MP4 (skipped when the
  same size is already there), then stages only `steps.parquet` and `meta.json`, last,
  with a `lance_videos` entry: `root`, `content_digest` and per camera `path`, `size` and
  `num_frames`. The committer checks the root, the path layout, and the size with one
  HEAD per file, takes frame counts from `meta.json`, and never reads an MP4; a mismatch
  rejects the delivery (REJECTED.json). The committer's bytes per episode drop to the
  parquet and `meta.json`.
- (b) go further: workers also write steps fragments and stage `fragments.json`. Only
  needed if the committer's parquet reads become the bottleneck.

Videos uploaded for a delivery that is then rejected or never staged are orphans.
`commit.orphan_videos(root, staging)` lists MP4s that no `recordings` version references,
no pending (unrejected) staged delivery claims, and that were last written more than
`VIDEO_UPLOAD_GRACE` (a day) ago, the window between a worker's upload and its `meta.json`;
it deletes only with `delete=True`. `commit.cleanup` and `gc_videos` take the staging root
too, so a pending direct-mode delivery keeps its videos however long it waits, and the
grace period is separate from `cleanup`'s version retention. Placing a video gives it a
fresh timestamp: one already in place (same size) is touched rather than skipped.

### Annotations

Vendor zarrs carry their language spans in the zarr `annotations` array (JSON
`{text, start_idx, end_idx}` per span, counted in zarr frames): checked on one episode each
of mecka (13 spans), scale (67), microagi v2 (6), sharpa_dex (6) and yam (1); the abc
episode had none. SQL `app.episodes.segments` (seconds) is filled only for 23,289 mecka rows,
and for the one sampled it matched the zarr spans one to one; abc and yam rows hold JSON
`null` there. So the zarr array is the source.

Lance training samples over task segments (#722) and reads spans from `annotations.lance`,
so the migration carries them: `zarr_to_lance` maps each span onto the recording clock
(frame i is steps row i, so episodes without `obs_rgb_timestamps_ns` map by index too; a
span ending at the last frame ends at the recording's `t_end_ns`) and writes them to
`meta.json` `annotations`, with unusable spans listed in `annotations_dropped`. An end
index past the last frame is clamped (the sampled yam span ends at 400 of 323 frames). After
each committed batch, `lance_commit` reviews the batch's spans like any change file and
commits them as one annotation change (`annotation_vendor` = the lab), tagged
`annotations-<UTC time>`. A span that fails review is reported and skipped; it never
rejects its recording. With `--replace`, a replaced recording's committed spans from the
delivery's vendors that the new delivery no longer carries are deleted in the same change.

## 5. Idempotency and resume

- Before each run the driver drops manifest rows that are already in `recordings`
  (`committed_ids`), already staged (`lance_staging/zarr/<vendor id>/meta.json` exists, so
  a half-staged delivery is redone), or already in the report (unless `--retry-failed`).
- The committer skips recording_ids it already has. Re-staging an existing recording is
  rejected with "already committed" unless `--replace` is given.
- A delivery that fails validation gets `REJECTED.json` (errors plus a file digest) and is
  skipped until its files change. The driver does not re-stage over a rejected delivery: the
  directory exists, so the episode is skipped. To retry one, delete its staging directory
  and rerun.
- Conversion failures (download, converter exception, id mismatch, or a task whose worker
  died past its retries) only go to the report, and the driver carries on. Nothing is
  written to R2 for them.

## 6. Validation gates

Per episode, enforced by the committer's `check_delivery`, so every delivery is checked:
steps rows == `num_frames`, timestamps strictly increasing, every MP4's frame count ==
`num_frames`, and lab, task_name and operator present. The committer report's
`annotations` lists, per batch, the spans committed and any review errors.

Per stage, run by hand before moving on:

1. **Frame-count reconciliation:** the sum of `num_frames` in `recordings` equals the sum
   over the committed manifest rows, and every report entry with `frames_mismatch` is
   explained. Zarr frame counts should equal SQL; a mismatch means SQL is stale.
2. **Numeric spot-check:** for ~1% of episodes (at least 20 per shard), load the zarr and
   the Lance rows and compare every numeric key exactly (float32), using
   `check_zarr_to_lance.py` from the converter branch. Also decode 10 random frames per
   camera and require PSNR against the zarr JPEG ≥ 33 dB. The reencode bench measured
   34–38 dB mean at CRF 36.
3. **Loader check:** `bench_loader` and `check_samples` on the new root, and a short
   training smoke run on one shard.
4. **Rejection budget:** stop the rollout if more than 1% of a stage is rejected or fails
   for a reason not seen before.

## 7. Cost and time estimate

Measured numbers this rests on:

- **Conversion speed:** EVA raw→Lance, 51 episodes, 49,000 frames, 3 AV1 streams at 360p:
  **38 episode-frames/s per process** (`lance_pilot/eva_deliveries/*/meta.json`
  `seconds_total`).
- **Zarr read and JPEG decode:** 66–92 ms per frame per camera on a single thread
  (`reencode_bench*/cost.csv`). AV1 encode (SVT preset 11, CRF 36) ran at ~10 frames/s
  per stream in the bench.
- **Output size:** EVA deliveries are ~19 KB/frame (3 streams). Aria is ~16 KB/frame
  (2 streams); 1-stream human is assumed at ~7 KB/frame.

Assumptions:

- (A1) A zarr episode converts at **~40 episode-frames/s per 8-vCPU task**. Human zarrs have
  one stream (cheaper to encode) but pay for JPEG decode.
- (A2) **~1,000 vCPU** of c6a/c7a in us-east-2, about $0.045/vCPU-h on demand and $0.015 spot.
- (A3) Output is ~3–4 TB for 364M frames.

| | zarr route now (499M frames, ~47 TB in) | everything incl. blocked (722M frames, 60 TB in) |
|---|---|---|
| compute | 12.5M task-s ≈ 28k vCPU-h → ~$1.25k on demand, ~$420 spot | ~40k vCPU-h → ~$1.8k / ~$600 |
| wall clock at 1,000 vCPU | ~28 h | ~40 h |
| R2 → EC2 reads | $0 egress; ~8M Class B GETs ≈ $3 | ~$5 |
| EC2 → R2 writes (AWS internet egress, $0.09/GB) | MP4 + parquet ~5 TB ≈ $450 (`--direct-videos` keeps it at one upload) | ~$600–900 |
| R2 storage of output | ~5 TB × $0.015/GB-mo ≈ $75/mo | ~$110/mo |

Notes:

- Run the workers in a public subnet with an internet gateway. Pulling 20–60 TB through a
  NAT gateway would add $0.045/GB, about $0.9k–2.7k.
- At these numbers the committer is the throughput risk, not the workers. Measure it in
  stage 3.
- A1 is the weakest number. The 100-episode stage replaces it with a measurement
  (`seconds_convert` and `seconds_download` in the report).

## 8. Rollout

Each stage reruns the planner, `--dry-run`s the driver, then runs it.

1. **1 episode:** `--limit 1 --shard 'rl2/eva_bimanual/*'` into a scratch root
   (`s3://rldb/lance_migration_test/`), then commit and run gates 1–3 by hand.
2. **100 episodes** across rl2 aria and eva shards (`--limit 100`), still on the test root.
   Measure per-worker frames/s, download MB/s and the committer's batch time; fix A1.
3. **One lab/embodiment**, rl2 eva_bimanual (2.6k episodes) then rl2 aria (2.6k), into the
   production root `s3://rldb/lance/`. Measure committer throughput and decide on the §4
   change before scaling.
4. **The vendor and robot shards:** sharpa_dex dexmate (115) and rl2/yam yam (2.5k) first,
   then scale (17k), mecka (42k), abc yam (37k) and microagi_v2 (186k). Commit cadence:
   batches of 256–512 (~1k tags), compaction after each shard, `cleanup` once the shard's
   gates pass.
5. Blocked sets follow once their blocker is fixed: operator backfill for microagi v1 and
   trace. Rerun the planner and the driver picks them up.

## 9. Raw path for Aria and EVA

Decision: Aria and EVA episodes are converted from their raw captures with `aria_to_lance`
and `eva_to_lance`, not from zarr. The raw converters keep every RGB frame (validity flags
instead of deleted rows), fit MANO on Aria, and flag EVA zero-pose dropouts.

### Raw inventory (full R2 listing, 2026-10-30)

`raw_v2/aria/` and `raw_v2/eva/` hold 53,979 objects, 24.2 TB. A capture is keyed by its
epoch ms, which is the SQL `episode_hash` (`timestamp_ms_to_episode_hash`):

- Aria: `{ms}.vrs` plus `mps_{ms}_vrs/hand_tracking/hand_tracking_results.csv` and
  `mps_{ms}_vrs/slam/closed_loop_trajectory.csv` (required), `eye_gaze/general_eye_gaze.csv`
  (optional). Metadata comes from SQL through the committer, so `{ms}.json` and
  `_metadata.json` are not downloaded. 2,575 VRS, 2,571 with MPS.
- EVA: `{ms}.hdf5` plus `{ms}_metadata.json`, from which `eva_to_lance` reads only the robot
  (embodiment); lab, operator, task, scene and objects come from the SQL row
  (`eva_to_lance --manifest`), as Aria's do. 3,254 captures, all complete. The HDF5s store raw 640x480 frames: ~3.5 GB per episode, ~2.8 MB/frame.
- Not indexed: `aria/backup/` (0.8 TB), `*/corrupted/` (0.3 TB), `.tmp/`, `raw/sieve/`,
  `raw_v2/test_*`, `raw_v2/{abc,yam}`.

Routed manifest (`--raw-listing`; `/coc/flash7/scratch/lance_pilot/migration_plan/raw/`):

| source | lab | embodiments | raw route | MPS | raw TB | SQL frames (M) |
|---|---|---|---:|---:|---:|---:|
| aria | rl2 | human_bimanual / left / right | 1,496 / 18 / 89 | all | 5.59 | 6.17 |
| aria | eth | human_bimanual / left / right | 225 / 179 / 136 | all | 3.12 | 4.13 |
| aria | song | human_bimanual / left / right | 109 / 120 / 51 | all | 0.60 | 0.91 |
| aria | wang | human_bimanual / left / right | 88 / 16 / 42 | all | 0.53 | 0.64 |
| eva | rl2 | eva_bimanual / eva_right_arm | 2,675 / 88 | n/a | 9.57 | 3.15 |
| **total** | | | **5,332** | | **19.4** | **15.0** |

Every Aria/EVA row that has a zarr also has complete raw, apart from one: rl2
`2026-10-01-23-11-33-000000` (legacy embodiment `aria`, no MPS hand tracking), which is
rejected either way. 51 EVA rows that have no zarr but do have raw become convertible.
Still lost: 116 EVA rows with neither zarr nor raw (2025-11-27 to 2026-01-26) and 3 Aria
rows with VRS but no SLAM output and no zarr. Also not migrated: 4 rl2 rows with the legacy
embodiment `eva` (they have raw), 2 Aria rows with no task, and 487 EVA captures (1.0 TB,
2025-11 to 2026-03) whose SQL rows are deleted.

### Planner

`plan_zarr_migration --raw-listing raw_listing.parquet` lists `raw_v2/{aria,eva}/` (or
reads a cached listing) and builds `raw_index` with one row per capture: `raw_complete`,
`mps_present`, `raw_uris`, `raw_bytes`. Each aria/eva manifest row with complete raw gets
`route = "raw"`, `converter` (module path) and `raw_uris`; the other aria/eva rows are
rejected (`no_raw` / `incomplete_raw`; 4 in v2), since `zarr_to_lance` refuses Aria and EVA
zarrs. Every other row has `route = "zarr"`. On the raw route, `no_zarr_path`, `zarr_processing_error` and
`no_frames` do not apply. Work units are `{shard}/{route}/{n}`, so a unit is run by exactly
one driver. `zarr_to_lance_ray` now only selects `route == "zarr"` rows.

`recording_id` is unchanged: `recording_id("aria"|"eva", epoch_ms)` on both routes, and it
matches what the raw converters compute. The 99 pilot recordings in `lance_v2` therefore
keep their ids, and a row that was first migrated from zarr is replaced by `--replace`.

### Worker flow (raw)

1. Download `raw_uris` to local NVMe into the layout the converter expects
   (`<dir>/{ms}.vrs`, `<dir>/mps_{ms}_vrs/...`; `<dir>/{ms}.hdf5`, `{ms}_metadata.json`).
   This is 3–7 GB per episode, so give each worker ≥ 20 GB of scratch and download one
   episode ahead at most.
2. Run the converter: `aria_to_lance --embodiment <row.embodiment>` (MANO on, 2400 iters),
   or `eva_to_lance --fps 30` with the manifest row as its SQL metadata.
3. Check `meta.recording_id == row.recording_id`. Stage to
   `lance_staging/{aria,eva}/{ms}/` (`commit.stage`, meta.json last; or `stage_direct` once
   it supports these sources), for `lance_commit --source aria|eva`.
4. Delete the local files and write a report line, as on the zarr path. Idempotency is the
   same as §5, keyed by `recording_id` and the staging directory.

### Cost (measured 2026-10-30, `migration_plan/raw/bench/`)

| step | measurement | per episode (avg 4.6k Aria / 1.1k EVA frames) |
|---|---|---|
| Aria numerics without MANO | 9 s / 4.6k frames | ~10 s |
| Aria MANO, A40 | 395 s / 4,587 frames, 2 hands = 0.043 s per frame-hand; 20 s per 512-frame chunk at 2,400 iters | ~6 min bimanual |
| Aria MANO, 8 vCPU | 116 s per 512-frame chunk = 0.23 s per frame-hand (5.3x the A40) | ~35 min bimanual |
| Aria undistort + 2 AV1 streams, 8 vCPU | 37 fps (pilot 48 episodes: 38 fps; a slow node gave 15) | ~2 min |
| EVA convert, 8 vCPU | 65 fps here, 38 fps over the 51-episode pilot | 20–30 s |
| peak RSS | 1.7 GB Aria with MANO, 0.8 GB EVA | |

MANO decides the cost. `torch` uses CUDA whenever a GPU is visible and otherwise runs on CPU.

| | Aria (2,569 ep., 11.8M frames, ~19.7M frame-hands) | EVA (2,763 ep., 3.15M frames) |
|---|---|---|
| MANO | ~235 A40 GPU-h, or ~1,240 8-vCPU task-h (~9.9k vCPU-h) | — |
| decode/encode + numerics | ~90 task-h (~710 vCPU-h) | ~15–25 task-h (~120–190 vCPU-h) |
| download | 9.85 TB | 9.57 TB |

On the rl2-lab A40s (free, since R2 has no egress charge) the whole raw path takes ~1 day
with 10–16 GPUs. On EC2 g5/g6 it costs ~$250–400 on demand. MANO on CPU instead is ~10k
vCPU-h, about $450 on demand or $150 spot. The fit loop is bound by iteration count, not
batch size (512 frames per chunk barely loads an A40), so `aria_to_lance --mano-batch`
(default 8,192) optimizes many 512-frame chunks together, with the loss still normalized
per chunk so results match, and the fit loop skins only the 5 fingertip vertices it needs
(`mano_joints_fn`) instead of all 778. On one 4,587-frame episode (2,400 iters, A40) MANO
dropped from 338 s to 52 s for both hands (6.5x), within 1e-3 mm of the old fit, which
cuts the GPU-h above to ~40.

### TODO (Ray driver and committer)

- Raw driver: `zarr_to_lance_ray` with a `route == "raw"` branch, or a new
  `raw_to_lance_ray`, that reads the manifest, downloads `raw_uris` and calls
  `aria_to_lance.convert` / `eva_to_lance.convert` in process. Aria tasks request
  `num_gpus` (fractional, e.g. 0.25, since one fit uses <2 GB) and fall back to CPU.
- `aria_to_lance_ray.py` (prototype) discovers recordings from a local directory and takes
  the embodiment from it. Switch it to manifest rows, or retire it in favour of the
  driver above.
- `stage_direct` (direct MP4 upload) for the aria and eva sources.
- MANO weights: `external_ckpts/` (with `external_ckpts/mano`) is gitignored, so a fresh
  checkout or container on a worker has no model files and the fit fails at the first
  episode. Ship them with the job (Ray `runtime_env` `working_dir`/`py_modules` excludes
  gitignored files, so upload them to R2 and download to `mano/_bundled_models` as
  `_default_mano_model_dir` expects, or bake them into the worker image) and check them at
  driver startup and in each worker before the first episode.

## Open decisions

1. Should microagi v1 (328k episodes, 197M frames, 11 TB, no operator, scene or rig) be
   migrated or dropped as superseded by v2? If it is migrated, who backfills operator? Trace
   (7.7k) needs an operator backfill in any case.
2. Resolved: yam_bimanual uses EVA_STEPS with end_effector `parallel_jaw_gripper`, and
   dexmate_bimanual has DEXMATE_STEPS with end_effector `dex_hand`.
3. Resolved: vendor and robot zarrs use `("zarr", episode_hash)`.
4. Tags: one tag per batch on every table (~1k tags for a full run) or only per stage, and
   the retention period passed to `cleanup`.
5. Whether to make the committer change in §4 (a) before stage 4.
6. The production root name, and whether the 99 pilot recordings in `lance_v2` are
   re-converted from zarr or kept from raw. Kept from raw is the default: their ids
   collide, so the zarr copies are skipped.
   With the raw path (§9) all of them are reconverted from raw anyway: keep the pilot rows,
   or `--replace` them?
7. Raw path: should the 4 rl2 rows with the legacy embodiment `eva` (which have raw) get
   their embodiment fixed in SQL? Are the 487 EVA captures whose SQL rows are deleted
   really dropped? (Resolved: no zarr fallback for Aria and EVA.)
8. Where MANO runs: the cluster A40s, EC2 GPUs, or EC2 CPU (about 10k vCPU-h before
   `--mano-batch`; re-measure on CPU with it).
