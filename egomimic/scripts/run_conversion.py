#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import ray
from cloudpathlib import S3Path
from ray.exceptions import OutOfMemoryError, RayTaskError, WorkerCrashedError

from egomimic.rldb.zarr.required_metadata import (
    WAIVABLE_FIELDS,
    check_writer_metadata,
    required_metadata_from_args,
)
from egomimic.scripts.aria_process.gen2_filter import Gen2FilterConfig
from egomimic.scripts.ray_helper import AriaRay, EmbodimentRay, EvaRay
from egomimic.utils.aws.aws_data_utils import (
    get_cloudpathlib_s3_client,
    load_env,
)
from egomimic.utils.aws.aws_sql import (
    TableRow,
    add_episode,
    create_default_engine,
    episode_hash_to_table_row,
    episode_table_to_df,
    episodes_with_path_prefix_or_error_marker,
    timestamp_ms_to_episode_hash,
    update_episode,
)
from egomimic.utils.type_utils import str2bool

PROCESSED_LOCAL_ROOT = Path(
    os.environ.get("PROCESSED_LOCAL_ROOT", "/home/ubuntu/processed")
).resolve()

LOG_ROOT = Path(
    os.environ.get(
        "CONVERSION_LOG_ROOT",
        str(PROCESSED_LOCAL_ROOT / "conversion_logs"),
    )
).resolve()


def ensure_path_ready(p: str | Path | S3Path, retries: int = 30) -> bool:
    if isinstance(p, str):
        if p.startswith("s3://"):
            s3_client = get_cloudpathlib_s3_client()
            p = S3Path(p, client=s3_client)
        else:
            p = Path(p)
    for _ in range(retries):
        try:
            if p.exists():
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _map_processed_local_to_remote(p: str | Path, processed_remote_prefix: str) -> str:
    """Map any path under PROCESSED_LOCAL_ROOT → PROCESSED_REMOTE_PREFIX/relative."""
    if not p:
        return ""
    p = Path(p).resolve()
    try:
        rel = p.relative_to(PROCESSED_LOCAL_ROOT)  # raises if not under root
    except Exception:
        return str(p)
    return (
        f"{processed_remote_prefix}/{rel.as_posix()}"
        if processed_remote_prefix
        else str(p)
    )


def infer_arm_from_row(row: TableRow) -> str:
    """
    Infer arm from SQL row.embodiment (e.g., 'aria_left', 'aria_right', 'aria_bimanual').
    Falls back to 'bimanual'.
    """
    emb = (row.embodiment or "").lower()
    if "left" in emb:
        return "left"
    if "right" in emb:
        return "right"
    if "bimanual" in emb:
        return "both"
    return "both"


def infer_task_name_from_row(row: TableRow) -> str:
    """
    Infer task name from SQL row.task (e.g., 'fold_clothes').
    Falls back to '', which required_metadata_row_error rejects.
    """
    return (row.task or "").strip()


_NULL_OBJECTS = frozenset({"", "none", "null"})


def parse_db_objects(raw) -> list[str]:
    """Object names from the DB objects column.

    Most rows hold a Postgres array literal ('{"pink cup",bag}'), some a plain
    comma-separated string; a driver may also hand back a list. None/NULL
    entries are dropped.
    """
    if isinstance(raw, (list, tuple)):
        entries = [str(e) for e in raw if e is not None]
    elif isinstance(raw, str):
        text = raw.strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        reader = csv.reader([text], skipinitialspace=True, escapechar="\\")
        entries = next(reader, [])
    else:
        entries = []
    return [e.strip() for e in entries if e.strip().lower() not in _NULL_OBJECTS]


def infer_required_metadata_from_row(row: TableRow) -> dict:
    """
    operator/lab/scene/objects for the converter, plus a missing_metadata list
    naming the waivable fields (scene, objects) the row left empty, so they are
    written as an explicit MissingMetadata(). An empty task, operator or lab is
    left as None and rejected by required_metadata_row_error. Plain data so it
    travels through Ray.
    """
    values = {
        "task_name": infer_task_name_from_row(row),
        "operator": (row.operator or "").strip(),
        "lab": (row.lab or "").strip(),
        "scene": (row.scene or "").strip(),
        "objects": parse_db_objects(row.objects),
    }
    missing = [field for field in WAIVABLE_FIELDS if not values[field]]
    del values["task_name"]
    return {**{k: v or None for k, v in values.items()}, "missing_metadata": missing}


def required_metadata_row_error(
    task_name: str | None, required_metadata: dict
) -> str | None:
    """First reason the row's metadata would be rejected by ZarrWriter, or None."""
    try:
        check_writer_metadata(
            required_metadata_from_args(
                SimpleNamespace(task_name=task_name, **required_metadata)
            )
        )
    except ValueError as e:
        return str(e)
    return None


def infer_task_description_from_row(row: TableRow) -> str:
    """
    Infer task description from SQL row.task_description (e.g., 'task_description').
    Falls back to ''.
    """
    return row.task_description or ""


def _demo_error_marker(recording_hash: str) -> str:
    return f"[recording {recording_hash}]"


def _demo_rows_of(
    engine, recording: TableRow, processed_remote_prefix: str
) -> dict[str, TableRow]:
    """Demo rows of a split recording, found by the parent link: demo zarrs
    live under <prefix>/<recording hash>/ and failed demos carry
    "[recording <hash>]" in zarr_processing_error. Never the recording row."""
    rows = episodes_with_path_prefix_or_error_marker(
        engine,
        f"{processed_remote_prefix}/{recording.episode_hash}/",
        _demo_error_marker(recording.episode_hash),
    )
    return {r.episode_hash: r for r in rows if r.episode_hash != recording.episode_hash}


def _retire(engine, rows, recording_hash: str) -> None:
    for row in rows:
        if not row.is_deleted:
            row.is_deleted = True
            update_episode(engine, row)
            print(
                f"[RETIRE] demo {row.episode_hash} from an earlier conversion of "
                f"{recording_hash}",
                flush=True,
            )


def _retire_demos_after_single_episode(
    engine, recording: TableRow, processed_remote_prefix: str, *, split_demos, frames
) -> None:
    """A Gen 2 recording converted as ONE episode replaces the demo rows of an
    earlier split. Only then: in split mode a single (ds, mp4, frames) result
    is a failure code, and a failed redo must leave the existing demos live."""
    if split_demos or frames is None or int(frames) <= 0:
        return
    _retire(
        engine,
        _demo_rows_of(engine, recording, processed_remote_prefix).values(),
        recording.episode_hash,
    )


def _record_demo_rows(
    engine, parent: TableRow, demos: list[dict], processed_remote_prefix: str
) -> None:
    """One SQL row per demo of a split Gen 2 recording.

    - Each demo row links to its recording through its zarr path
      (<prefix>/<recording hash>/<demo hash>.zarr; see _demo_rows_of).
    - New demo rows copy the recording row's labels and is_eval; existing ones
      are updated in place (conversion fields only, so is_eval / eval_*
      survive a redo). Deleted demo rows are never resurrected and count as
      done.
    - Demo rows from an earlier conversion that are not in this one are
      retired (is_deleted), so a redo cannot duplicate training data.
    - A hash held by a row that is not one of this recording's demos is a
      clash: that demo is not written and the recording is flagged.
    - Last, the recording row is retired from conversion via
      zarr_processing_error ("split into N demos ..."; the schema has no
      status column), which launch() already skips. Until then every step is
      idempotent, so a failure part-way is simply redone on the next run.
    """
    prefix = f"{processed_remote_prefix}/{parent.episode_hash}"
    siblings = _demo_rows_of(engine, parent, processed_remote_prefix)
    done, live, clashes = [], [], []
    for demo in demos:
        frames = int(demo["frames"])
        existing = episode_hash_to_table_row(engine, demo["episode_hash"])
        if existing is not None and existing.episode_hash not in siblings:
            clashes.append(demo["episode_hash"])
            print(
                f"[ERR] demo hash {demo['episode_hash']} of {parent.episode_hash} "
                "belongs to another episode; not written",
                flush=True,
            )
            continue
        if existing is not None and existing.is_deleted:
            print(f"[SKIP] demo {existing.episode_hash} was deleted", flush=True)
            done.append(existing.episode_hash)
            continue
        row = existing or TableRow(
            episode_hash=demo["episode_hash"],
            operator=parent.operator,
            lab=parent.lab,
            task=parent.task,
            embodiment=parent.embodiment,
            rig_name=parent.rig_name,
            task_description=parent.task_description,
            scene=parent.scene,
            objects=parent.objects,
            is_eval=parent.is_eval,
        )
        row.num_frames = frames
        if frames > 0:
            row.zarr_processed_path = f"{prefix}/{Path(demo['ds_path']).name}"
            row.zarr_mp4_path = (
                f"{prefix}/{Path(demo['mp4_path']).name}"
                if demo["mp4_path"] and demo["mp4_path"] != "None"
                else ""
            )
            row.zarr_processing_error = ""
            done.append(row.episode_hash)
            live.append(row.episode_hash)
        else:
            row.zarr_processed_path = ""
            row.zarr_mp4_path = ""
            reason = "Upload Failed" if frames == -2 else "Zero Frames"
            row.zarr_processing_error = (
                f"{reason} {_demo_error_marker(parent.episode_hash)}"
            )
        if existing is None:
            add_episode(engine, row)
        else:
            update_episode(engine, row)
        print(
            f"[OK] demo {row.episode_hash} of {parent.episode_hash}: "
            f"num_frames={frames}, zarr_processed_path={row.zarr_processed_path}",
            flush=True,
        )

    current = {d["episode_hash"] for d in demos}
    _retire(
        engine,
        [r for h, r in siblings.items() if h not in current],
        parent.episode_hash,
    )

    parent.num_frames = sum(
        int(d["frames"]) for d in demos if d["episode_hash"] in live
    )
    parent.zarr_processed_path = ""
    parent.zarr_mp4_path = ""
    if clashes:
        parent.zarr_processing_error = f"Demo hash clash: {', '.join(clashes)}"
    elif len(done) == len(demos):
        parent.zarr_processing_error = (
            f"split into {len(demos)} demos: {demos[0]['episode_hash']} .. "
            f"{demos[-1]['episode_hash']}"
        )
    else:
        parent.zarr_processing_error = ""
    update_episode(engine, parent)


def _is_oom_exception(e: Exception) -> bool:
    if isinstance(e, OutOfMemoryError):
        return True
    if isinstance(e, (RayTaskError, WorkerCrashedError)):
        s = str(e).lower()
        return (
            ("outofmemory" in s)
            or ("out of memory" in s)
            or ("oom" in s)
            or ("killed" in s)
        )
    s = str(e).lower()
    return ("outofmemory" in s) or ("out of memory" in s) or ("oom" in s)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def isatty(self) -> bool:
        return False


# --- Ray task ----------------------------------------------------------------
def submit_convert(embodiment_ray: EmbodimentRay, size: str, **kwargs):
    embodiment_ray_cls = embodiment_ray.__class__
    if size == "small":
        num_cpus = embodiment_ray.num_cpus_small
        resources = embodiment_ray.resources_small

    else:
        num_cpus = embodiment_ray.num_cpus_big
        resources = embodiment_ray.resources_big

    return (
        ray.remote(embodiment_ray_cls.convert_one_bundle)
        .options(num_cpus=num_cpus, resources=resources)
        .remote(**kwargs)
    )


# --- Driver ------------------------------------------------------------------
def launch(
    embodiment: str,
    dry: bool = False,
    skip_if_done: bool = False,
    episode_hashes: list[str] | None = None,
    gen2_source: str = "mps",
    split_demos: bool = True,
    gen2_filter: dict | None = None,
):
    embodiment_ray = None
    if embodiment in ("aria", "aria_gen2"):
        embodiment_ray = AriaRay(
            PROCESSED_LOCAL_ROOT,
            LOG_ROOT,
            device=embodiment,
            gen2_source=gen2_source,
            split_demos=split_demos,
            gen2_filter=gen2_filter,
        )
    elif embodiment == "eva":
        embodiment_ray = EvaRay(
            PROCESSED_LOCAL_ROOT,
            LOG_ROOT,
        )
    else:
        raise ValueError(f"Invalid embodiment: {embodiment}")

    engine = create_default_engine()
    pending: Dict[ray.ObjectRef, Dict[str, Any]] = {}

    benchmark_rows = []

    df = episode_table_to_df(engine)

    for name, args in embodiment_ray.iter_bundles():
        # IMPORTANT: episode_hash is TEXT in DB; do not cast to int
        episode_key = timestamp_ms_to_episode_hash(int(name))
        row = df[df["episode_hash"] == episode_key]
        if len(row) == 1:
            row = row.iloc[0]
        elif len(row) > 1:
            print("[WARNING] Duplicate episode hash")
        else:
            row = None

        if not episode_key:
            print(f"[SKIP] {name}: could not parse episode_hash from stem", flush=True)
            continue

        if episode_hashes is not None and episode_key not in episode_hashes:
            print(
                f"[SKIP] {name}: episode_key '{episode_key}' not in provided episode_hashes list",
                flush=True,
            )
            continue

        if row is None:
            print(f"[SKIP] {name}: no matching row in SQL (app.episodes)", flush=True)
            continue

        processed_path = (row.zarr_processed_path or "").strip()
        if skip_if_done and len(processed_path) > 0:
            print(
                f"[SKIP] {name}: already has zarr_processed_path='{processed_path}'",
                flush=True,
            )
            continue

        if row.zarr_processing_error != "":
            print(
                f"[INFO] skipping episode hash: {row.episode_hash} due to zarr processing error",
                flush=True,
            )
            continue

        if row.is_deleted:
            print(f"[SKIP] {name}: episode marked as deleted in SQL", flush=True)
            continue

        print(f"[INFO] processing {name}: episode_key={episode_key}", flush=True)

        arm = infer_arm_from_row(row)
        task_name = infer_task_name_from_row(row)
        task_description = infer_task_description_from_row(row)
        # TODO: add dry run, right now the script does not work with this dry run code
        # out_dir = PROCESSED_LOCAL_ROOT
        # if dry:
        #     ds_path = (PROCESSED_LOCAL_ROOT / dataset_name).resolve()
        #     stem = name
        #     mp4_candidate = PROCESSED_LOCAL_ROOT / f"{stem}_video.mp4"

        #     mapped_ds = _map_processed_local_to_remote(
        #         ds_path, embodiment_ray.processed_remote_prefix
        #     )
        #     mapped_mp4 = _map_processed_local_to_remote(
        #         mp4_candidate, embodiment_ray.processed_remote_prefix
        #     )

        #     print(
        #         f"[DRY] {name}: arm={arm} | out_dir={out_dir}/{dataset_name}\n"
        #         f"      would write to SQL:\n"
        #         f"        zarr_processed_path={mapped_ds}\n"
        #         f"        zarr_mp4_path={mapped_mp4}",
        #         flush=True,
        #     )
        #     continue

        args["arm"] = arm
        args["task_name"] = task_name or None
        args["required_metadata"] = infer_required_metadata_from_row(row)
        args["task_description"] = task_description
        # Reject here, not after a full extraction, and record why.
        metadata_error = required_metadata_row_error(
            args["task_name"], args["required_metadata"]
        )
        if metadata_error is not None:
            print(f"[SKIP] {name}: {metadata_error}", flush=True)
            if not dry and (
                table_row := episode_hash_to_table_row(engine, episode_key)
            ):
                table_row.zarr_processing_error = f"Metadata: {metadata_error}"
                update_episode(engine, table_row)
            continue
        start_time = time.time()
        ref = submit_convert(embodiment_ray, "small", **args)
        pending[ref] = {
            "episode_key": episode_key,
            "start_time": start_time,
            "size": "small",
            "args": args,
        }

    if dry or not pending:
        return

    # Collect and update SQL (with OOM retry on BIG)
    while pending:
        done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
        ref = done_refs[0]
        info = pending.pop(ref)

        episode_key = info["episode_key"]
        start_time = info["start_time"]
        duration_sec = time.time() - start_time

        row = episode_hash_to_table_row(engine, episode_key)
        if row is None:
            print(
                f"[WARN] Episode {episode_key}: row disappeared before update?",
                flush=True,
            )
            continue

        try:
            result = ray.get(ref)  # can throw (OOM, index error, etc.)
            if isinstance(result, list):
                try:
                    _record_demo_rows(
                        engine, row, result, embodiment_ray.processed_remote_prefix
                    )
                except Exception as e:  # idempotent: redone on the next run
                    print(
                        f"[ERR] demo rows for {episode_key} incomplete, will be "
                        f"redone: {type(e).__name__}: {e}",
                        flush=True,
                    )
                continue
            ds_path, mp4_path, frames = result
            if embodiment == "aria_gen2":
                _retire_demos_after_single_episode(
                    engine,
                    row,
                    embodiment_ray.processed_remote_prefix,
                    split_demos=embodiment_ray.split_demos,
                    frames=frames,
                )

            row.num_frames = int(frames) if frames is not None else -1
            if row.num_frames > 0:
                row.zarr_processed_path = _map_processed_local_to_remote(
                    ds_path, embodiment_ray.processed_remote_prefix
                )
                row.zarr_mp4_path = _map_processed_local_to_remote(
                    mp4_path, embodiment_ray.processed_remote_prefix
                )
                row.zarr_processing_error = ""
            elif row.num_frames == -2:
                row.zarr_processed_path = ""
                row.zarr_mp4_path = ""
                row.zarr_processing_error = "Upload Failed"
            elif row.num_frames == -1:
                row.zarr_processed_path = ""
                row.zarr_mp4_path = ""
                row.zarr_processing_error = "Zero Frames"
            else:
                row.zarr_processed_path = ""
                row.zarr_mp4_path = ""
                row.zarr_processing_error = "Conversion Failed Unhandled Error"

            update_episode(engine, row)
            print(
                f"[OK] Updated SQL for {episode_key}: "
                f"zarr_processed_path={row.zarr_processed_path}, num_frames={row.num_frames}, "
                f"duration_sec={duration_sec:.2f}",
                flush=True,
            )

            if row.num_frames > 0 and row.zarr_processed_path:
                benchmark_rows.append(
                    {
                        "episode_key": episode_key,
                        "processed_path": row.zarr_processed_path,
                        "mp4_path": row.zarr_mp4_path,
                        "num_frames": row.num_frames,
                        "duration_sec": duration_sec,
                    }
                )

        except Exception as e:
            # If OOM on small, retry once on big
            if _is_oom_exception(e) and info.get("size") == "small":
                print(
                    f"[OOM] Episode {episode_key} failed on SMALL. Retrying on BIG...",
                    flush=True,
                )
                args = info["args"]
                ref2 = submit_convert(embodiment_ray, "big", **args)
                pending[ref2] = {
                    **info,
                    "start_time": time.time(),
                    "size": "big",
                }
                continue

            print(
                f"[FAIL] Episode {episode_key} task failed ({info.get('size', '?')}): "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

            # mark failed in SQL (so skip-if-done won't think it's done)
            row.num_frames = -1
            row.zarr_processed_path = ""
            row.zarr_mp4_path = ""
            row.zarr_processing_error = f"{type(e).__name__}: {e}"
            try:
                update_episode(engine, row)
                print(
                    f"[FAIL] Marked SQL failed for {episode_key} (cleared zarr_processed_path)",
                    flush=True,
                )
            except Exception as ee:
                print(
                    f"[ERR] SQL update failed for failed episode {episode_key}: {ee}",
                    flush=True,
                )

    if benchmark_rows:
        timing_file = Path(f"{embodiment}_conversion_timings.csv")
        file_exists = timing_file.exists()
        fieldnames = [
            "episode_key",
            "processed_path",
            "mp4_path",
            "num_frames",
            "duration_sec",
        ]
        try:
            with timing_file.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for bench_row in benchmark_rows:
                    writer.writerow(bench_row)
            print(
                f"[BENCH] wrote {len(benchmark_rows)} entries → {timing_file.resolve()}",
                flush=True,
            )
        except Exception as e:
            print(f"[ERR] Failed to write benchmark CSV {timing_file}: {e}", flush=True)


# --- CLI ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--embodiment", type=str, required=True, choices=["aria", "aria_gen2", "eva"]
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Skip episodes that already have a zarr_processed_path in SQL",
    )
    p.add_argument(
        "--ray-address", default="auto", help="Ray cluster address (default: auto)"
    )
    p.add_argument(
        "--episode-hash",
        nargs="+",
        dest="episode_hashes",
        help="Episode hashes to process. Separate multiple hashes with spaces.",
    )
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--gen2-source",
        choices=["mps", "on_device"],
        default="mps",
        help="aria_gen2 only: convert from MPS outputs (mps_*_vrs/ next to the "
        "VRS, required) or from the VRS's on-device hand tracking / VIO / gaze.",
    )
    p.add_argument(
        "--split-demos",
        type=str2bool,
        default=True,
        help="aria_gen2 only: one episode (zarr + SQL row) per demo, split at "
        "resets and the opening calibration.",
    )
    p.add_argument(
        "--gen2-filter-config",
        type=Path,
        default=None,
        help="aria_gen2 only: YAML of Gen2FilterConfig overrides.",
    )
    p.add_argument(
        "--working-dir",
        default="/home/ubuntu/EgoVerse",
        help="Repo checkout shipped to ray workers as runtime_env working_dir "
        "(only used with --debug). Lets a non-default checkout (e.g. a worktree "
        "on a feature branch) drive the conversion fleet.",
    )
    p.add_argument(
        "--py-modules",
        nargs="+",
        default=None,
        help="Extra python package dirs shipped to workers via runtime_env "
        "py_modules (e.g. mano + patched chumpy + trimesh for the MANO "
        "keypoint conversion).",
    )
    args = p.parse_args()
    # Validate the Gen 2 filter config here: an error inside a Ray task would
    # be recorded as a failed conversion and never retried.
    gen2_filter = (
        dataclasses.asdict(Gen2FilterConfig.from_yaml(args.gen2_filter_config))
        if args.gen2_filter_config
        else None
    )

    env_vars = {}
    load_env()
    for k in [
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_SESSION_TOKEN",  # optional
        "R2_ENDPOINT_URL",  # optional; include if your helper expects it
    ]:
        v = os.environ.get(k)
        if v:
            env_vars[k] = v

    if args.debug:
        runtime_env = {
            "working_dir": args.working_dir,
            "excludes": [
                "**/.git/**",
                "external/openpi/third_party/aloha/**",
                "**/*.pack",
                "**/__pycache__/**",
                "external/openpi/**",
                "temp_dir/**",
            ],
        }
    else:
        runtime_env = {}
    if args.py_modules:
        runtime_env["py_modules"] = list(args.py_modules)
    runtime_env["env_vars"] = env_vars
    ray.init(address=args.ray_address, runtime_env=runtime_env)
    launch(
        embodiment=args.embodiment,
        dry=args.dry_run,
        skip_if_done=args.skip_if_done,
        episode_hashes=args.episode_hashes,
        gen2_source=args.gen2_source,
        split_demos=args.split_demos,
        gen2_filter=gen2_filter,
    )


if __name__ == "__main__":
    main()
