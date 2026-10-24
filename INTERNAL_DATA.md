# EgoVerse Data: Internal Notes (RL2)

*For RL2 team members: ingest, the episode database, and the internal converters.*

External contributors follow [CONTRIBUTING_DATA.md](CONTRIBUTING_DATA.md), the
vendor guide. Everything in that guide (episode hash, zarr format, metadata
rules, validation) applies to internal data too; this file covers only what
vendors cannot or need not do. A bare `§N` refers to a section of this file.

---

## Table of Contents

1. [Credentials](#1-credentials)
2. [Episode Database](#2-episode-database)
3. [Ingesting a Vendor Delivery](#3-ingesting-a-vendor-delivery)
4. [Internal Conversion](#4-internal-conversion)
5. [Training Data Configs](#5-training-data-configs)
6. [Internal S3 Paths and Uploads](#6-internal-s3-paths-and-uploads)
7. [Internal Checklist](#7-internal-checklist)

---

## 1. Credentials

RL2 members need AWS credentials for the episode registry (PostgreSQL via
Secrets Manager) in addition to the R2 bucket credentials:

```bash
aws configure
# AccessKeyId / SecretAccessKey: ask the consortium lead
# Default region: us-east-2
bash egomimic/utils/aws/setup_secret.sh
# Writes ~/.egoverse_env with R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
# AWS_ENDPOINT_URL_S3, SECRETS_ARN, etc.
```

Verify:

```python
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.aws.aws_sql import create_default_engine

load_env()
engine = create_default_engine()   # should print: Tables in schema 'app': ['episodes']
```

---

## 2. Episode Database

Every episode is registered in the PostgreSQL `app.episodes` table, the
authoritative index used by all download and training tooling.

### 2.1 Schema

The authoritative schema is the `TableRow` dataclass in
[egomimic/utils/aws/aws_sql.py](egomimic/utils/aws/aws_sql.py). Refer to that
file for the exact fields, defaults, and types.

Key field notes:
- `episode_hash`: PRIMARY KEY, must match the `.zarr` directory name exactly (CONTRIBUTING_DATA.md §4).
- `operator`: stable per-collector ID; a pseudonymous ID is fine (CONTRIBUTING_DATA.md §6.3).
- `lab`: who the data came from; a full match of `^[a-z0-9][a-z0-9_-]*$`. Once set, do not change it (used in filters and partitioning). Rows that fail this rule are rejected by `run_conversion` (§4.2).
- `task`: the `task_name` that groups related episodes. Before adding a new name, check the existing tasks via [`sql_tutorial.ipynb`](egomimic/scripts/tutorials/sql_tutorial.ipynb) (`df.groupby("task").size()`) and reuse one if the episode fits. A new name is a short, stable, lowercase category (e.g. `fold_clothes`, `object_in_container`), not a one-off trial description.
- `objects`: stored as text, usually a Postgres array literal (`{"pink cup",bag}`); `run_conversion` parses it into a list (§4.2).
- `embodiment`: one of the strings in CONTRIBUTING_DATA.md §9.
- `rig_name`: the physical capture rig (e.g. `aria_gen1`, `eva`). Optional.

### 2.2 Inserting a Row

```python
from egomimic.utils.aws.aws_sql import TableRow, add_episode, create_default_engine
from egomimic.utils.aws.aws_data_utils import load_env

load_env()
engine = create_default_engine()

row = TableRow(
    episode_hash   = "2026-03-15-14-22-10-000000",
    operator       = "op_7",
    lab            = "rl2",
    task           = "fold_clothes",
    embodiment     = "human_bimanual",
    rig_name       = "aria_gen1",
    task_description = "folding a 2T baby shirt on a blue table",
    scene          = "kitchen_A",
    objects        = "baby_shirt_2T",
    num_frames     = 2712,
)

add_episode(engine, row)
```

`add_episode` raises `RuntimeError` on a duplicate `episode_hash`. Check for
collisions (`episode_hash_to_table_row(engine, hash)` returns `None`) before
inserting.

### 2.3 Updating a Row After Upload

```python
from egomimic.utils.aws.aws_sql import update_episode

row.zarr_processed_path = "s3://rldb/processed_v3/aria/2026-03-15-14-22-10-000000.zarr"
row.num_frames = 2712
update_episode(engine, row)
```

---

## 3. Ingesting a Vendor Delivery

Vendors upload validated zarr episodes and previews to
`s3://rldb/processed_v3/<lab>/` and raw recordings to
`s3://rldb/raw/<lab>/<vendor_recording_id>/` (CONTRIBUTING_DATA.md §10.1). They send no delivery note
or manifest. On ingest:

1. Run `python -m egomimic.rldb.zarr.validate` on the episodes. Episodes with
   `scene` or `objects` stored as `null` (waived with `MissingMetadata()`), and
   `dexmate_bimanual` episodes with `null` intrinsics, pass with
   `--allow-missing` and are accepted. Episodes without a `schema_version`
   (written before the required metadata) only get warnings for metadata.
2. Map each vendor-proposed `task_name` to an existing task (§2.1), or add a new
   task name if none fits.
3. Insert one DB row per episode (§2.2), with `lab`, `operator`, `scene`,
   `objects` and the mapped `task` taken from the episode's `zarr.attrs`, and
   `zarr_processed_path` / `zarr_mp4_path` set to the uploaded paths.
4. Add a training data config for the vendor if it is new (§5).

---

## 4. Internal Conversion

### 4.1 Converter CLIs

`aria_to_zarr.py` and `eva_to_zarr.py` take the required metadata on the command
line through `egomimic.rldb.zarr.required_metadata.add_required_metadata_args`:
`--task-name`, `--operator`, `--lab`, `--scene`, `--objects`, and a repeatable
`--missing-metadata FIELD` to write `scene` or `objects` as `MissingMetadata()`
(`task_name`, `operator` and `lab` cannot be waived). None of them
has a usable default: a missing flag without a waiver fails when the episode is
written. `--save-mp4` writes the CONTRIBUTING_DATA.md §6.5 preview next to the `.zarr`.

### 4.2 Batch conversion from the DB (`run_conversion.py`)

```bash
python egomimic/scripts/run_conversion.py --embodiment aria   # or eva
```

It walks the raw bundles, looks up each episode's DB row, and converts it on Ray.
The required metadata comes from the row:

- `task` → `task_name`; `operator`, `lab`, `scene` as stored (stripped).
- `objects` is parsed from the Postgres array literal or comma-separated text;
  `None` / `NULL` entries are dropped.
- An empty `scene` or `objects` is written as `MissingMetadata()`. An empty
  `task`, `operator` or `lab` is a metadata error (below).
- A row whose metadata the writer would reject (a placeholder such as
  `"unknown"` or `"debug"`, a non-identifier `lab`, ...) is skipped before
  extraction, and `zarr_processing_error` is set to `Metadata: <reason>`. Fix
  the row and clear `zarr_processing_error` to have it picked up again; rows
  with a non-empty `zarr_processing_error` are always skipped.

### 4.3 Disabled converters

`egomimic/scripts/mecka_process/mecka_to_zarr.py` (`MeckaDatasetConverter` and
its CLI), `external/scale/scripts/sfs_to_egoverse_zarr.py`
(`convert_task_to_zarr`) and `external/scale/scripts/sfs_zarr_pipeline.py` raise
`NotImplementedError`. Tracking: #738 (mecka), #739 (Scale). They predate the required metadata and mandatory
intrinsics. Re-enabling either one means passing `intrinsics={"front_1": 3x4 K}`,
`task_name`, `operator`, `lab`, `scene` and `objects` to
`ZarrWriter.create_and_write` as real arguments (`metadata_override` may not
set keys the writer owns). Reconverted mecka zarrs also need the fixed
left-wrist convention (`Human.get_transform_list(fix_left_wrist_convention=...)`).
The mecka pose helpers (`left_wrist_convention` and friends) still work and are
used by training.

---

## 5. Training Data Configs

Every human contributor uses the single `Human` embodiment (CONTRIBUTING_DATA.md §9.1). Its per-vendor
structural choices are explicit arguments from the data config:

- `Human.get_keymap(keymap_mode="cartesian"|"keypoints", has_head_pose=<bool>, include_aria_keypoints=<bool>)`
- `Human.get_transform_list(mode="cartesian_6d"|"keypoints_headframe_ypr"|..., stride=<int>)`

To onboard a new vendor, add a data config under
[`egomimic/hydra_configs/data/`](egomimic/hydra_configs/data/) whose `key_map` /
`transform_list` point at those with the args the data needs. Copy `aria.yaml`
(head-mounted, `stride: 3`) or `scale.yaml` (`stride: 1`).

- `has_head_pose=False` if the data has no `obs_head_pose`; `stride` is the
  action-chunk stride (`3` for ~30 fps egocentric, `1` for already-downsampled data).
- **Robots** subclass `Embodiment` directly and keep their own
  intrinsics/extrinsics and pipeline — see `Eva` in
  [`egomimic/rldb/embodiment/eva.py`](egomimic/rldb/embodiment/eva.py).
- **Dexmate** (`dexmate_bimanual`, SharpaDex: a Dexmate robot with two Sharpa
  Wave hands) has a keypoints mode only:
  [`Dexmate`](egomimic/rldb/embodiment/dexmate.py) reads the palm poses and
  MANO-layout hand keypoints and re-expresses them in `obs_head_link_pose` (all
  tracks are in the robot base frame). Train with `train_zarr_dexmate_keypoints`
  (`data/dexmate_keypoints.yaml`, `model/hpt_bc_keypoints_dexmate.yaml`, 140-wide
  actions). It has no video evaluator, because the episodes have no camera
  calibration. Joints, commands and tactile are stored but not loaded. The
  arrays an episode must hold are in
  [`dexmate_rules.py`](egomimic/rldb/zarr/dexmate_rules.py), which the
  validator applies; there is no Pi mode for Dexmate.
- For data annotated through Scale AI, `ScaleAnnotationDatasetFilter`
  ([`egomimic/rldb/filters.py`](egomimic/rldb/filters.py)) keeps only episodes
  with completed Scale annotations; it needs `SCALE_API_KEY`.

---

## 6. Internal S3 Paths and Uploads

Internal conversions upload under an embodiment prefix, set with
`PROCESSED_REMOTE_PREFIX` (defaults in `ray_helper.py`:
`s3://rldb/processed_v3/aria` and `s3://rldb/processed_v3/eva`). The preview MP4
goes next to the zarr under the same prefix.

Bulk upload with Ray:

```python
import ray
from egomimic.utils.aws.aws_data_utils import upload_dir_to_s3, load_env

ray.init()

@ray.remote
def upload_one(local_zarr_path: str, s3_prefix: str):
    load_env()
    upload_dir_to_s3(local_zarr_path, bucket="rldb", prefix=s3_prefix)

tasks = [
    upload_one.remote(f"/local/processed/{h}.zarr", f"processed_v3/aria/{h}.zarr")
    for h in episode_hashes
]
ray.get(tasks)
```

Download for local work with
[`egomimic/scripts/data_download/sync_s3.py`](egomimic/scripts/data_download/sync_s3.py)
(`--local-dir`, optional named `--filters`).

---

## 7. Internal Checklist

In addition to the vendor checklist (CONTRIBUTING_DATA.md §12):

- [ ] Episode hash is not already in the DB (`episode_hash_to_table_row(engine, hash)` returns `None`).
- [ ] DB row inserted, with `task` mapped to an existing task name (§3).
- [ ] `embodiment` and `task` in the DB row match `zarr.attrs` (`embodiment`, mapped `task_name`).
- [ ] `zarr_processed_path` and `zarr_mp4_path` point at the uploaded files.
- [ ] `num_frames` in the DB row matches `total_frames` in `zarr.attrs`.
- [ ] `sync_s3.py` with an appropriate filter can download and open the episode.
