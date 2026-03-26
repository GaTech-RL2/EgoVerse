#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict

import ray
from cloudpathlib import S3Path
from ray.exceptions import OutOfMemoryError, RayTaskError, WorkerCrashedError

from egomimic.scripts.ray_helper import AriaRay, EmbodimentRay, EvaRay
from egomimic.utils.aws.aws_data_utils import (
    get_cloudpathlib_s3_client,
    load_env,
)
from egomimic.utils.aws.aws_sql import (
    TableRow,
    create_default_engine,
    episode_hash_to_table_row,
    episode_table_to_df,
    timestamp_ms_to_episode_hash,
    update_episode,
)

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
    Infer arm from SQL row.robot_name (e.g., 'aria_left', 'aria_right', 'aria_bimanual').
    Falls back to 'bimanual'.
    """
    emb = (row.robot_name or "").lower()
    if "left" in emb:
        return "left"
    if "right" in emb:
        return "right"
    if "bimanual" in emb:
        return "both"
    return "both"


def infer_task_name_from_row(row: TableRow) -> str:
    """
    Infer task name from SQL row.task_name (e.g., 'task_name').
    Falls back to ''.
    """
    return row.task or "unknown"


def infer_task_description_from_row(row: TableRow) -> str:
    """
    Infer task description from SQL row.task_description (e.g., 'task_description').
    Falls back to ''.
    """
    return row.task_description or ""


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
):
    embodiment_ray = None
    if embodiment == "aria":
        embodiment_ray = AriaRay(
            PROCESSED_LOCAL_ROOT,
            LOG_ROOT,
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
        args["task_name"] = task_name
        args["task_description"] = task_description
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
            ds_path, mp4_path, frames = ray.get(
                ref
            )  # can throw (OOM, index error, etc.)

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
    p.add_argument("--embodiment", type=str, required=True, choices=["aria", "eva"])
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
        action="append",
        dest="episode_hashes",
        help="Episode hash to process. Can be specified multiple times to process multiple episodes.",
    )
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

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
            "working_dir": "/home/ubuntu/EgoVerse",
            "excludes": [
                "**/.git/**",
                "external/openpi/third_party/aloha/**",
                "**/*.pack",
                "**/__pycache__/**",
                "external/openpi/**",
            ],
        }
    else:
        runtime_env = {}
    runtime_env["env_vars"] = env_vars
    ray.init(address=args.ray_address, runtime_env=runtime_env)
    launch(
        embodiment=args.embodiment,
        dry=args.dry_run,
        skip_if_done=args.skip_if_done,
        episode_hashes=args.episode_hashes,
    )


if __name__ == "__main__":
    main()
