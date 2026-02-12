#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import ray
from cloudpathlib import S3Path
from ray.exceptions import OutOfMemoryError, RayTaskError, WorkerCrashedError

from eva_helper import zarr_job

from egomimic.utils.aws.aws_data_utils import (
    get_boto3_s3_client,
    get_cloudpathlib_s3_client,
    load_env,
    upload_dir_to_s3,
)
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_hash_to_table_row,
    episode_table_to_df,
    timestamp_ms_to_episode_hash,
    update_episode,
)

RAW_REMOTE_PREFIX = os.environ.get("RAW_REMOTE_PREFIX", "s3://rldb/raw_v2/eva").rstrip("/")
PROCESSED_LOCAL_ROOT = Path(os.environ.get("PROCESSED_LOCAL_ROOT", "/home/ubuntu/processed")).resolve()
PROCESSED_REMOTE_PREFIX = os.environ.get("PROCESSED_REMOTE_PREFIX", "s3://rldb/processed_v3/eva").rstrip("/")
BUCKET = os.environ.get("BUCKET", "rldb")
LOG_ROOT = Path(
    os.environ.get(
        "EVA_CONVERSION_LOG_ROOT",
        str(PROCESSED_LOCAL_ROOT / "eva_conversion_logs"),
    )
).resolve()

DEFAULT_EXTRINSICS_KEY = "x5Dec13_2"


def ensure_path_ready(p: str | Path, retries: int = 30) -> bool:
    p = Path(p)
    for _ in range(retries):
        try:
            if p.exists():
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _map_processed_local_to_remote(p: str | Path) -> str:
    if not p:
        return ""
    p = Path(p).resolve()
    try:
        rel = p.relative_to(PROCESSED_LOCAL_ROOT)
    except Exception:
        return str(p)
    return (
        f"{PROCESSED_REMOTE_PREFIX}/{rel.as_posix()}"
        if PROCESSED_REMOTE_PREFIX
        else str(p)
    )


def _load_extrinsics_key_from_json(meta_json: Path) -> str:
    if not meta_json.is_file():
        return DEFAULT_EXTRINSICS_KEY

    try:
        obj = json.loads(meta_json.read_text())
    except Exception:
        return DEFAULT_EXTRINSICS_KEY

    if isinstance(obj, dict) and "extrinsics_key" in obj:
        val = obj["extrinsics_key"]
        if isinstance(val, str) and len(val) > 0:
            return val

    return DEFAULT_EXTRINSICS_KEY


def iter_hdf5_bundles_s3(root_s3: str) -> Iterator[Tuple[S3Path, str]]:
    """Walk R2 for *.hdf5 files; load extrinsics_key from sidecar JSON if present."""
    s3_client = get_cloudpathlib_s3_client()
    root = S3Path(root_s3, client=s3_client)
    for hdf5 in sorted(root.glob("*.hdf5"), key=lambda p: p.name):
        name = hdf5.stem
        meta_json_s3 = root / f"{name}_metadata.json"
        extrinsics_key = DEFAULT_EXTRINSICS_KEY
        try:
            if meta_json_s3.exists():
                obj = json.loads(meta_json_s3.read_text())
                if isinstance(obj, dict) and isinstance(obj.get("extrinsics_key"), str) and obj["extrinsics_key"]:
                    extrinsics_key = obj["extrinsics_key"]
        except Exception:
            pass
        yield hdf5, extrinsics_key


def infer_arm_from_robot_name(robot_name: str | None) -> str:
    s = (robot_name or "").lower()
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    if "bimanual" in s or "both" in s:
        return "both"
    return "both"


def _load_episode_key(name: str) -> str | None:
    try:
        return datetime.fromtimestamp(float(name) / 1000.0, timezone.utc).strftime(
            "%Y-%m-%d-%H-%M-%S-%f"
        )
    except Exception:
        return None


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


def _parse_s3_uri(uri: str, *, default_bucket: str | None = None) -> tuple[str, str]:
    """
    Parse s3 URI or key prefix.
      - "s3://bucket/prefix" -> ("bucket", "prefix")
      - "prefix" -> (default_bucket, "prefix")
    """
    uri = (uri or "").strip()
    if uri.startswith("s3://"):
        rest = uri[len("s3://"):]
        bucket, _, key_prefix = rest.partition("/")
        return bucket, key_prefix.strip("/")
    if default_bucket is None:
        raise ValueError(f"Expected s3://... but got '{uri}' and no default_bucket provided")
    return default_bucket, uri.strip("/")


def convert_one_bundle_impl(
    data_h5_s3: str,
    out_dir: str,
    s3_processed_dir: str,
    dataset_name: str,
    arm: str,
    description: str,
    extrinsics_key: str,
) -> tuple[str, str, int]:
    s3_client = get_boto3_s3_client()
    hdf5_s3 = S3Path(data_h5_s3)
    stem = hdf5_s3.stem

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{stem}-{uuid.uuid4().hex[:8]}.log"

    tmp_dir = Path.home() / "temp_eva_processing" / f"{stem}-{uuid.uuid4().hex[:6]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log_fh:
        tee_out = _Tee(sys.stdout, log_fh)
        tee_err = _Tee(sys.stderr, log_fh)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"[LOG] {stem}: {log_path}", flush=True)

            raw_bucket, raw_prefix = _parse_s3_uri(RAW_REMOTE_PREFIX, default_bucket=BUCKET)
            raw_root = S3Path(RAW_REMOTE_PREFIX)

            rel = hdf5_s3.relative_to(raw_root).as_posix()
            t_key = f"{raw_prefix.rstrip('/')}/{rel}".strip("/")
            local_hdf5 = tmp_dir / hdf5_s3.name
            try:
                s3_client.download_file(raw_bucket, t_key, str(local_hdf5))
            except Exception as e:
                print(f"[ERR] aws download failed for {data_h5_s3}: {e}", flush=True)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return "", "", -1

            ds_parent = Path(out_dir)
            ds_parent.mkdir(parents=True, exist_ok=True)
            ds_path = ds_parent / dataset_name

            try:
                print(
                    f"[INFO] Converting: {stem} → {ds_path} (arm={arm}, extrinsics_key={extrinsics_key})",
                    flush=True,
                )
                job_kwargs = dict(
                    raw_path=str(local_hdf5),
                    output_dir=str(ds_parent),
                    dataset_name=dataset_name,
                    arm=arm,
                    description=description or "",
                    extrinsics_key=extrinsics_key,
                )
                zarr_path, mp4_path = zarr_job(**job_kwargs)

                frames = -1
                zarr_store_path = zarr_path
                info = zarr_store_path / "zarr.json"
                print(f"[DEBUG] Zarr metadata path: {info}", flush=True)
                if info.exists():
                    try:
                        meta = json.loads(info.read_text())
                        print(f"[DEBUG] Zarr metadata keys: {list(meta.keys())}", flush=True)
                        frames = int(meta.get("attributes", {}).get("total_frames", -1))
                    except Exception as e:
                        print(f"[ERR] Failed to parse zarr metadata {info}: {e}", flush=True)
                        frames = -1
                else:
                    print(f"[ERR] Zarr metadata not found: {info}", flush=True)
                path_for_sql = f"{PROCESSED_REMOTE_PREFIX}/{stem}.zarr"

                try:
                    out_bucket, out_prefix = _parse_s3_uri(s3_processed_dir, default_bucket=BUCKET)
                    ds_s3_prefix = f"{out_prefix.rstrip('/')}/{dataset_name}.zarr".strip("/")
                    upload_dir_to_s3(str(zarr_store_path), out_bucket, prefix=ds_s3_prefix)
                    shutil.rmtree(str(zarr_store_path), ignore_errors=True)
                    print(f"[CLEANUP] Removed local zarr store: {zarr_store_path}", flush=True)
                    mp4_obj = Path(mp4_path)
                    mp4_rel = mp4_obj.resolve().relative_to(PROCESSED_LOCAL_ROOT).as_posix()
                    mp4_s3_key = f"{out_prefix.rstrip('/')}/{mp4_rel}".strip("/")
                    s3_client.upload_file(str(mp4_obj), out_bucket, mp4_s3_key)
                    mp4_obj.unlink(missing_ok=True)
                    print(f"[CLEANUP] Removed local mp4: {mp4_obj}", flush=True)
                except Exception as e:
                    print(f"[ERR] Failed to upload {zarr_store_path} to S3: {e}", flush=True)
                    return zarr_path, mp4_path, -2

                return zarr_path, mp4_path, frames

            except Exception as e:
                err_msg = f"[FAIL] {stem}: {e}\n{traceback.format_exc()}"
                print(err_msg, flush=True)
                return zarr_path, "", -1
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)


@ray.remote(num_cpus=2, resources={"eva_small": 1})
def convert_one_bundle_small(*args, **kwargs):
    return convert_one_bundle_impl(*args, **kwargs)


@ray.remote(num_cpus=8, resources={"eva_big": 1})
def convert_one_bundle_big(*args, **kwargs):
    return convert_one_bundle_impl(*args, **kwargs)


def launch(
    dry: bool = False,
    skip_if_done: bool = False,
    episode_hashes: list[str] | None = None,
):
    engine = create_default_engine()
    pending: Dict[ray.ObjectRef, Dict[str, Any]] = {}
    benchmark_rows = []

    df = episode_table_to_df(engine)

    for hdf5_s3, extrinsics_key in iter_hdf5_bundles_s3(RAW_REMOTE_PREFIX):
        name = hdf5_s3.stem
        episode_key = _load_episode_key(name)

        row_match = df[df["episode_hash"] == episode_key]
        if len(row_match) == 1:
            row = row_match.iloc[0]
        elif len(row_match) > 1:
            print("[WARNING] Duplicate episode hash", flush=True)
            row = row_match.iloc[0]
        else:
            row = None

        if not episode_key:
            print(f"[SKIP] {name}: could not derive DB episode key", flush=True)
            continue

        if episode_hashes is not None and episode_key not in episode_hashes:
            print(
                f"[SKIP] {name}: episode_key '{episode_key}' not in provided episode_hashes list",
                flush=True,
            )
            print(f"[SKIP] {name}: no matching row in SQL (app.episodes)", flush=True)
            continue

        if row is None:
            print(f"[SKIP] {name}: no matching row in SQL (app.episodes)", flush=True)
            continue

        processed_path = (row.zarr_processed_path or "").strip()
        processing_error = row.zarr_processing_error
        path_field_name = "zarr_processed_path"
       

        if skip_if_done and len(processed_path) > 0:
            print(f"[SKIP] {name}: already has {path_field_name}='{processed_path}'", flush=True)
            continue

        if processing_error != "":
            print(
                f"[INFO] skipping episode hash: {row.episode_hash} due to processing error",
                flush=True,
            )
            continue

        if row.is_deleted:
            print(f"[SKIP] {name}: episode marked as deleted in SQL", flush=True)
            continue

        print(f"[INFO] processing {name}: episode_key={episode_key}", flush=True)

        arm = infer_arm_from_robot_name(getattr(row, "robot_name", None))
        dataset_name = hdf5_s3.stem
        dataset_name = timestamp_ms_to_episode_hash(name)
        out_dir = PROCESSED_LOCAL_ROOT
        s3out_dir = PROCESSED_REMOTE_PREFIX
        description = row.task_description or ""

        if dry:
            ds_path = (PROCESSED_LOCAL_ROOT / dataset_name).resolve()
            mp4_candidate = PROCESSED_LOCAL_ROOT / f"{name}_video.mp4"
            # Zarr stored flat under eva: prefix/<stem>.zarr
            mapped_ds = f"{PROCESSED_REMOTE_PREFIX}/{name}.zarr"
            mapped_mp4 = _map_processed_local_to_remote(mp4_candidate)
            path_field_name = "zarr_processed_path"
            mp4_field_name = "zarr_mp4_path"
            print(
                f"[DRY] {name}: arm={arm} | out_dir={out_dir}/{dataset_name}\n"
                f"      desc-bytes={len(description.encode('utf-8'))}\n"
                f"      extrinsics_key={extrinsics_key}\n"
                f"      would write to SQL:\n"
                f"        {path_field_name}={mapped_ds}\n"
                f"        {mp4_field_name}={mapped_mp4}",
                flush=True,
            )
            continue

        args_tuple = (
            str(hdf5_s3),
            str(out_dir),
            str(s3out_dir),
            dataset_name,
            arm,
            description,
            extrinsics_key,
        )

        start_time = time.time()
        ref = convert_one_bundle_small.remote(*args_tuple)
        pending[ref] = {
            "episode_key": episode_key,
            "dataset_name": dataset_name,
            "start_time": start_time,
            "size": "small",
            "args": args_tuple,
        }

    if dry or not pending:
        return

    while pending:
        done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
        ref = done_refs[0]
        info = pending.pop(ref)

        episode_key = info["episode_key"]
        start_time = info["start_time"]
        duration_sec = time.time() - start_time

        row = episode_hash_to_table_row(engine, episode_key)
        if row is None:
            print(f"[WARN] Episode {episode_key}: row disappeared before update?", flush=True)
            continue

        try:
            ds_path, mp4_path, frames = ray.get(ref)

            row.num_frames = int(frames) if frames is not None else -1
            mapped_ds = _map_processed_local_to_remote(ds_path)
            mapped_mp4 = _map_processed_local_to_remote(mp4_path)
            
            if row.num_frames > 0:
                row.zarr_processed_path = mapped_ds
                row.zarr_mp4_path = mapped_mp4
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
            path_value = row.zarr_processed_path
           

            update_episode(engine, row)
            path_field_name = "zarr_processed_path"
            print(
                f"[OK] Updated SQL for {episode_key}: "
                f"{path_field_name}={path_value}, num_frames={row.num_frames}, "
                f"duration_sec={duration_sec:.2f}",
                flush=True,
            )

            if row.num_frames > 0 and path_value:
                mp4_val = row.zarr_mp4_path
                benchmark_rows.append(
                    {
                        "episode_key": episode_key,
                        "processed_path": path_value,
                        "mp4_path": mp4_val,
                        "num_frames": row.num_frames,
                        "duration_sec": duration_sec,
                    }
                )

        except Exception as e:
            if _is_oom_exception(e) and info.get("size") == "small":
                print(
                    f"[OOM] Episode {episode_key} failed on SMALL. Retrying on BIG...",
                    flush=True,
                )
                args_tuple = info["args"]
                ref2 = convert_one_bundle_big.remote(*args_tuple)
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

            row.num_frames = -1
            error_msg = f"{type(e).__name__}: {e}"
            path_field_name = "zarr_processed_path"
            
            row.zarr_mp4_path = ""
            row.zarr_processed_path = ""
            row.zarr_processing_error = error_msg
            
            try:
                update_episode(engine, row)
                print(
                    f"[FAIL] Marked SQL failed for {episode_key} (cleared {path_field_name})",
                    flush=True,
                )
            except Exception as ee:
                print(f"[ERR] SQL update failed for failed episode {episode_key}: {ee}", flush=True)

    if benchmark_rows:
        timing_file = Path("./eva_conversion_timings.csv")
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Skip episodes that already have a processed_path in SQL",
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
        "R2_SESSION_TOKEN",
        "R2_ENDPOINT_URL",
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
        dry=args.dry_run,
        skip_if_done=args.skip_if_done,
        episode_hashes=args.episode_hashes,
    )


if __name__ == "__main__":
    main()
