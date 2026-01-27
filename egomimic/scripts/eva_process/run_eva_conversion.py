#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Tuple

import ray
import traceback

from eva_helper import lerobot_job

from egomimic.utils.aws.aws_sql import (
    TableRow,
    create_default_engine,
    episode_hash_to_table_row,
    update_episode,
)

RAW_ROOT = Path("/mnt/raw")
PROCESSED_ROOT = Path("/mnt/processed")
PROCESSED_LOCAL_ROOT = Path(os.environ.get("PROCESSED_LOCAL_ROOT", "/mnt/processed")).resolve()
PROCESSED_REMOTE_PREFIX = os.environ.get("PROCESSED_REMOTE_PREFIX", "s3://rldb/processed_v2/eva").rstrip("/")

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
    return f"{PROCESSED_REMOTE_PREFIX}/{rel.as_posix()}" if PROCESSED_REMOTE_PREFIX else str(p)


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


def iter_hdf5_bundles(root: Path) -> Iterator[Tuple[Path, str]]:
    for data in sorted(root.glob("*.hdf5")):
        name = data.stem

        meta_json = root / f"{name}_metadata.json"
        extrinsics_key = _load_extrinsics_key_from_json(meta_json)

        yield data, extrinsics_key


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


@ray.remote(num_cpus=12)
def convert_one_bundle(
    data_h5: str,
    out_dir: str,
    dataset_name: str,
    arm: str,
    description: str,
    extrinsics_key: str,
) -> tuple[str, str, int]:
    stem = Path(data_h5).stem
    tmp_dir = Path.home() / "temp_eva_processing" / f"{stem}-{uuid.uuid4().hex[:6]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        Path(data_h5).resolve(strict=True),
    ]

    for t in targets:
        if not ensure_path_ready(t):
            print(f"[ERR] missing {t}", flush=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return "", "", -1
        link = tmp_dir / t.name
        try:
            os.symlink(t, link, target_is_directory=t.is_dir())
        except FileExistsError:
            pass

    ds_parent = Path(out_dir)
    ds_parent.mkdir(parents=True, exist_ok=True)
    ds_path = ds_parent / dataset_name

    try:
        print(
            f"[INFO] Converting: {stem} → {ds_path} (arm={arm}, extrinsics_key={extrinsics_key})",
            flush=True,
        )
        lerobot_job(
            raw_path=str(tmp_dir),
            output_dir=str(ds_parent),
            dataset_name=dataset_name,
            arm=arm,
            description=description or "",
            extrinsics_key=extrinsics_key,
        )

        frames = -1
        info = ds_path / "meta/info.json"
        if info.exists():
            try:
                meta = json.loads(info.read_text())
                frames = int(meta.get("total_frames", -1))
            except Exception:
                frames = -1

        mp4_candidates = list(ds_parent.glob(f"*{stem}*_video.mp4")) + list(
            ds_path.glob("**/*_video.mp4")
        )
        mp4_str = str(mp4_candidates[0]) if mp4_candidates else ""

        return str(ds_path), mp4_str, frames

    except Exception as e:
        err_msg = f"[FAIL] {stem}: {e}\n{traceback.format_exc()}"
        print(err_msg, flush=True)
        return str(ds_path), "", -1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def launch(dry: bool = False, skip_if_done: bool = False):
    engine = create_default_engine()
    pending: dict = {}

    for data_h5, extrinsics_key in iter_hdf5_bundles(RAW_ROOT):
        name = data_h5.stem
        episode_key = _load_episode_key(name)
        if not episode_key:
            print(f"[SKIP] {name}: could not derive DB episode key")
            continue

        row = episode_hash_to_table_row(engine, episode_key)
        if row is None:
            print(f"[SKIP] {name}: no matching row in SQL (app.episodes)")
            continue

        processed_path = (row.processed_path or "").strip()
        if skip_if_done and len(processed_path) > 0:
            print(f"[SKIP] {name}: already has processed_path='{processed_path}'")
            continue
        
        if row.is_deleted:
            print(f"[SKIP] {name}: episode marked as deleted in SQL")
            continue

        arm = infer_arm_from_robot_name(getattr(row, "robot_name", None))
        dataset_name = episode_key
        out_dir = PROCESSED_ROOT
        description = row.task_description or ""

        if dry:
            ds_path = (PROCESSED_ROOT / dataset_name).resolve()
            mp4_candidate = PROCESSED_ROOT / f"{name}_video.mp4"
            mapped_ds = _map_processed_local_to_remote(ds_path)
            mapped_mp4 = _map_processed_local_to_remote(mp4_candidate)
            print(
                f"[DRY] {name}: arm={arm} | out_dir={out_dir}/{dataset_name}\n"
                f"      desc-bytes={len(description.encode('utf-8'))}\n"
                f"      extrinsics_key={extrinsics_key}\n"
                f"      would write to SQL:\n"
                f"        processed_path={mapped_ds}\n"
                f"        mp4_path={mapped_mp4}"
            )
            continue

        ref = convert_one_bundle.remote(
            str(data_h5),
            str(out_dir),
            dataset_name,
            arm,
            description,
            extrinsics_key,
        )
        pending[ref] = (episode_key, dataset_name)

    if dry or not pending:
        return

    while pending:
        done_refs, _ = ray.wait(list(pending), num_returns=1)
        ref = done_refs[0]
        ds_path, mp4_path, frames = ray.get(ref)
        episode_key, _dataset_name = pending.pop(ref)

        row = episode_hash_to_table_row(engine, episode_key)
        if row is None:
            print(f"[WARN] Episode {episode_key}: row disappeared before update?")
            continue

        row.num_frames = int(frames) if frames is not None else -1
        if row.num_frames > 0:
            row.processed_path = _map_processed_local_to_remote(ds_path)
            row.mp4_path = _map_processed_local_to_remote(mp4_path)
            row.processing_error = ""
        else:
            row.processed_path = ""
            row.mp4_path = ""
            row.processing_error = "Zero Frames"

        try:
            update_episode(engine, row)
            print(
                f"[OK] Updated SQL for {episode_key}: "
                f"processed_path={row.processed_path}, num_frames={row.num_frames}"
            )
        except Exception as e:
            print(f"[ERR] SQL update failed for {episode_key}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Skip episodes that already have a processed_path in SQL",
    )
    args = p.parse_args()

    ray.init(address="auto")
    launch(dry=args.dry_run, skip_if_done=args.skip_if_done)


if __name__ == "__main__":
    main()
