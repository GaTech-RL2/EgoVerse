import contextlib
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from abc import abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Tuple

from cloudpathlib import S3Path

from egomimic.scripts.aria_process.aria_to_zarr import main as aria_main
from egomimic.scripts.eva_process.eva_to_zarr import main as eva_main
from egomimic.utils.aws.aws_data_utils import (
    delete_s3_key_if_exists,
    delete_s3_prefix,
    get_boto3_s3_client,
    get_cloudpathlib_s3_client,
    s3_sync_to_local,
    upload_dir_to_s3,
)


def ensure_path_ready(p: S3Path, retries: int = 30) -> bool:
    if not isinstance(p, S3Path):
        raise ValueError(f"Expected S3Path, got {type(p)}")
    for _ in range(retries):
        try:
            if p.exists():
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _parse_s3_uri(uri: str, *, default_bucket: str | None = None) -> tuple[str, str]:
    """
    Parse s3 URI or key prefix.
      - "s3://bucket/prefix" -> ("bucket", "prefix")
      - "prefix" -> (default_bucket, "prefix")
    """
    uri = (uri or "").strip()
    if uri.startswith("s3://"):
        rest = uri[len("s3://") :]
        bucket, _, key_prefix = rest.partition("/")
        return bucket, key_prefix.strip("/")
    if default_bucket is None:
        raise ValueError(
            f"Expected s3://... but got '{uri}' and no default_bucket provided"
        )
    return default_bucket, uri.strip("/")


def _cleanup_existing_processed_outputs(
    *,
    bucket: str,
    zarr_prefix: str,
    mp4_key: str | None,
) -> None:
    deleted_zarr_objects = delete_s3_prefix(bucket, zarr_prefix)
    if deleted_zarr_objects > 0:
        print(
            f"[CLEANUP] Deleted existing remote zarr prefix s3://{bucket}/{zarr_prefix} "
            f"({deleted_zarr_objects} objects)",
            flush=True,
        )
    else:
        print(
            f"[CLEANUP] No existing remote zarr objects at s3://{bucket}/{zarr_prefix}",
            flush=True,
        )

    if mp4_key:
        deleted_mp4 = delete_s3_key_if_exists(bucket, mp4_key)
        if deleted_mp4:
            print(
                f"[CLEANUP] Deleted existing remote mp4 s3://{bucket}/{mp4_key}",
                flush=True,
            )
        else:
            print(
                f"[CLEANUP] No existing remote mp4 at s3://{bucket}/{mp4_key}",
                flush=True,
            )

    print("[CLEANUP] Remote cleanup complete; continuing upload", flush=True)


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


class EmbodimentRay:
    def __init__(
        self,
        processed_local_root: Path,
        log_root: Path,
    ):
        self.processed_local_root = processed_local_root
        self.log_root = log_root
        self.num_cpus_small = 2
        self.num_cpus_big = 8

    @abstractmethod
    def iter_bundles(self, root_s3: str):
        pass

    @abstractmethod
    def convert_one_bundle(self, **kwargs):
        pass


class AriaRay(EmbodimentRay):
    def __init__(
        self,
        processed_local_root: Path,
        log_root: Path,
    ):
        super().__init__(
            processed_local_root,
            log_root,
        )
        self.raw_remote_prefix = os.environ.get(
            "RAW_REMOTE_PREFIX", "s3://rldb/raw_v2/aria"
        ).rstrip("/")
        self.processed_remote_prefix = os.environ.get(
            "PROCESSED_REMOTE_PREFIX", "s3://rldb/processed_v3/aria"
        ).rstrip("/")
        self.bucket = os.environ.get("BUCKET", "rldb")
        self.resources_small = {"aria_small": 1}  # TODO: change to aria_small
        self.resources_big = {"aria_big": 1}  # TODO: change to aria_big
        self.num_cpus_small = 2
        self.num_cpus_big = 8
        self.log_root = log_root

    def iter_bundles(self):
        """
        root_s3: like "s3://rldb/raw_v2/aria/"
        Returns S3Path objects (cloudpathlib), not local filesystem paths.

        Uses a single `root.walk(...)` traversal and avoids per-path `.exists()` / `.is_dir()`.
        """
        s3_client = get_cloudpathlib_s3_client()
        root = S3Path(self.raw_remote_prefix, client=s3_client)

        vrs_by_name: dict[str, S3Path] = {}
        has_json: set[str] = set()
        has_hand: set[str] = set()
        has_slam: set[str] = set()

        # Prefer topdown so we can prune recursion aggressively (don’t enumerate huge mps trees).
        try:
            walker = root.walk(topdown=True)  # cloudpathlib often mirrors os.walk
            can_prune = True
        except TypeError:
            walker = root.walk()
            can_prune = False  # can’t reliably prune, but still single API surface

        for dirpath, dirnames, filenames in walker:
            # Figure out depth relative to `root`
            try:
                rel = dirpath.relative_to(root)
                rel_str = rel.as_posix()
                rel_parts = () if rel_str in (".", "") else rel.parts
            except Exception:
                rel_parts = ()

            depth = len(rel_parts)

            if depth == 0:
                # Root-level files: *.vrs and *.json
                for fn in filenames:
                    if fn.endswith(".vrs"):
                        name = fn[:-4]
                        vrs_by_name[name] = dirpath / fn
                    elif fn.endswith(".json"):
                        has_json.add(fn[:-5])

                # Only descend into potential mps dirs
                if can_prune:
                    dirnames[:] = [
                        d
                        for d in dirnames
                        if d.startswith("mps_") and d.endswith("_vrs")
                    ]

            elif depth == 1:
                # We’re inside something like mps_{name}_vrs/
                d0 = rel_parts[0]
                if d0.startswith("mps_") and d0.endswith("_vrs"):
                    name = d0[len("mps_") : -len("_vrs")]
                    # If these prefixes exist (i.e., have objects under them), they should appear as dirnames.
                    if "hand_tracking" in dirnames:
                        has_hand.add(name)
                    if "slam" in dirnames:
                        has_slam.add(name)

                # We don’t need to enumerate anything deeper.
                if can_prune:
                    dirnames[:] = []

            else:
                if can_prune:
                    dirnames[:] = []

        # Match original ordering: sort by vrs filename
        for filename in sorted(vrs_by_name, key=lambda n: vrs_by_name[n].name):
            missing = []
            if filename not in has_json:
                missing.append("json")
            if filename not in has_hand:
                missing.append("hand_tracking")
            if filename not in has_slam:
                missing.append("slam")
            if missing:
                print(f"[MISSING] {filename}: has VRS but MISSING {missing}", flush=True)
                continue
            if True:
                vrs = vrs_by_name[filename]
                jsonf = root / f"{filename}.json"
                mps_dir = root / f"mps_{filename}_vrs"
                # arm, task_name, task_description are inferred from row in run_converion.py
                args = {
                    "processed_local_root": str(self.processed_local_root),
                    "processed_remote_prefix": self.processed_remote_prefix,
                    "bucket": self.bucket,
                    "raw_remote_prefix": self.raw_remote_prefix,
                    "log_root": str(self.log_root),
                    "vrs": str(vrs),
                    "jsonf": str(jsonf),
                    "mps_dir": str(mps_dir),
                    "out_dir": str(self.processed_local_root),
                    "fps": 30,
                    "chunk_timesteps": 100,
                    "save_mp4": True,
                    "image_compressed": False,
                }
                name = vrs_by_name[filename].stem
                yield name, args

    @staticmethod
    def convert_one_bundle(
        processed_local_root: str,
        processed_remote_prefix: str,
        bucket: str,
        raw_remote_prefix: str,
        log_root: str,
        vrs: str,
        jsonf: str,
        mps_dir: str,
        out_dir: str,
        arm: str,
        fps: int,
        task_name: str,
        task_description: str,
        chunk_timesteps: int,
        image_compressed: bool,
        save_mp4: bool,
    ) -> tuple[str, str, int]:
        """
        Perform conversion for a single episode.
        Returns (ds_path, mp4_path, total_frames).
        • ds_path: dataset folder path
        • mp4_path: per-episode MP4 ('' if not created)
        • total_frames: -1 if unknown/failure
        """
        processed_local_root = Path(processed_local_root)
        processed_remote_prefix = processed_remote_prefix.rstrip("/")
        bucket = bucket.rstrip("/")
        log_root = Path(log_root)
        raw_remote_prefix = raw_remote_prefix.rstrip("/")
        s3_client = get_cloudpathlib_s3_client()
        boto3_client = get_boto3_s3_client()
        vrs = S3Path(vrs, client=s3_client) if isinstance(vrs, str) else vrs
        jsonf = S3Path(jsonf, client=s3_client) if isinstance(jsonf, str) else jsonf
        mps_dir = (
            S3Path(mps_dir, client=s3_client) if isinstance(mps_dir, str) else mps_dir
        )

        stem = vrs.stem
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{stem}-{uuid.uuid4().hex[:8]}.log"

        tmp_dir = Path.home() / "temp_mps_processing" / f"{stem}-{uuid.uuid4().hex[:6]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", encoding="utf-8") as log_fh:
            tee_out = _Tee(sys.stdout, log_fh)
            tee_err = _Tee(sys.stderr, log_fh)
            with (
                contextlib.redirect_stdout(tee_out),
                contextlib.redirect_stderr(tee_err),
            ):
                print(f"[LOG] {stem}: {log_path}", flush=True)
                targets = [
                    vrs,
                    jsonf,
                    mps_dir,
                ]

                raw_bucket, raw_prefix = _parse_s3_uri(
                    raw_remote_prefix, default_bucket=bucket
                )
                raw_root = S3Path(raw_remote_prefix, client=s3_client)

                for t in targets:
                    if not ensure_path_ready(t):
                        print(f"[ERR] missing {t}", flush=True)
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        return "", "", -1
                    link = tmp_dir / t.name
                    # `t` is an S3Path; compute relative key under RAW_REMOTE_PREFIX.
                    rel = t.relative_to(raw_root).as_posix()
                    t_key = f"{raw_prefix.rstrip('/')}/{rel}".strip("/")

                    try:
                        if t.is_dir():
                            s3_sync_to_local(raw_bucket, t_key, str(link))
                        else:
                            boto3_client.download_file(raw_bucket, t_key, str(link))
                    except Exception as e:
                        print(f"[ERR] aws copy failed for {t}: {e}", flush=True)
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        return "", "", -1

                ds_parent = Path(out_dir)
                ds_parent.mkdir(parents=True, exist_ok=True)
                vrs_path = tmp_dir / vrs.name

                try:
                    zarr_path, mp4_path = AriaRay.zarr_job(
                        raw_path=str(vrs_path),
                        output_dir=str(ds_parent),
                        arm=arm,
                        fps=fps,
                        task_name=task_name,
                        task_description=task_description,
                        chunk_timesteps=chunk_timesteps,
                        image_compressed=image_compressed,
                        save_mp4=save_mp4,
                    )
                    frames = -1
                    zarr_store_path = zarr_path
                    info = zarr_store_path / "zarr.json"
                    print(f"[DEBUG] Zarr metadata path: {info}", flush=True)
                    if info.exists():
                        try:
                            meta = json.loads(info.read_text())
                            print(
                                f"[DEBUG] Zarr metadata keys: {list(meta.keys())}",
                                flush=True,
                            )
                            frames = int(
                                meta.get("attributes", {}).get("total_frames", -1)
                            )
                        except Exception as e:
                            print(
                                f"[ERR] Failed to parse zarr metadata {info}: {e}",
                                flush=True,
                            )
                            frames = -1
                    else:
                        print(f"[ERR] Zarr metadata not found: {info}", flush=True)
                        frames = -1

                    try:
                        out_bucket, out_prefix = _parse_s3_uri(
                            processed_remote_prefix, default_bucket=bucket
                        )
                        zarr_filename = Path(zarr_path).stem
                        ds_s3_prefix = (
                            f"{out_prefix.rstrip('/')}/{zarr_filename}.zarr".strip("/")
                        )
                        mp4_s3_key = None
                        if mp4_path:
                            mp4_s3_key = (
                                f"{out_prefix.rstrip('/')}/{Path(mp4_path).name}".strip(
                                    "/"
                                )
                            )
                        _cleanup_existing_processed_outputs(
                            bucket=out_bucket,
                            zarr_prefix=ds_s3_prefix,
                            mp4_key=mp4_s3_key,
                        )
                        upload_dir_to_s3(
                            str(zarr_store_path), out_bucket, prefix=ds_s3_prefix
                        )
                        shutil.rmtree(str(zarr_store_path), ignore_errors=True)
                        print(
                            f"[CLEANUP] Removed local zarr store: {zarr_store_path}",
                            flush=True,
                        )
                        if mp4_path:
                            boto3_client.upload_file(
                                str(mp4_path), out_bucket, mp4_s3_key
                            )
                            Path(mp4_path).unlink(missing_ok=True)
                            print(
                                f"[CLEANUP] Removed local mp4: {mp4_path}", flush=True
                            )
                    except Exception as e:
                        print(
                            f"[ERR] Failed to upload {zarr_store_path} to S3: {e}",
                            flush=True,
                        )
                        return "", "", -2

                    return str(zarr_path), str(mp4_path), frames

                except Exception as e:
                    err_msg = f"[FAIL] {stem}: {e}\n{traceback.format_exc()}"
                    print(err_msg, flush=True)
                    return "", "", -1
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def zarr_job(
        raw_path: str | Path,
        output_dir: str | Path,
        arm: str,
        fps: int = 30,
        task_name: str = "",
        task_description: str = "",
        chunk_timesteps: int = 100,
        image_compressed: bool = False,
        save_mp4: bool = True,
    ) -> None:
        args = SimpleNamespace(
            raw_path=raw_path,
            output_dir=output_dir,
            arm=arm,
            fps=fps,
            task_name=task_name,
            task_description=task_description,
            chunk_timesteps=chunk_timesteps,
            image_compressed=image_compressed,
            save_mp4=save_mp4,
            debug=False,
        )

        return aria_main(args)


class EvaRay(EmbodimentRay):
    def __init__(
        self,
        processed_local_root: Path,
        log_root: Path,
    ):
        super().__init__(
            processed_local_root,
            log_root,
        )
        self.raw_remote_prefix = os.environ.get(
            "RAW_REMOTE_PREFIX", "s3://rldb/raw_v2/eva"
        ).rstrip("/")
        self.processed_remote_prefix = os.environ.get(
            "PROCESSED_REMOTE_PREFIX", "s3://rldb/processed_v3/eva"
        ).rstrip("/")
        self.bucket = os.environ.get("BUCKET", "rldb")
        self.resources_small = {"eva_small": 1}
        self.resources_big = {"eva_big": 1}

    def iter_bundles(self) -> Iterator[Tuple[S3Path, str]]:
        """Walk R2 for *.hdf5 files."""
        s3_client = get_cloudpathlib_s3_client()
        root = S3Path(self.raw_remote_prefix, client=s3_client)
        for hdf5 in sorted(root.glob("*.hdf5"), key=lambda p: p.name):
            name = hdf5.stem
            # Unused for now
            # meta_json_s3 = root / f"{name}_metadata.json"
            # try:
            #     if meta_json_s3.exists():
            #         obj = json.loads(meta_json_s3.read_text())
            # except Exception:
            #     pass
            # taskname and arm are inferred from row in run_converion.py
            args = {
                "processed_local_root": str(self.processed_local_root),
                "processed_remote_prefix": str(self.processed_remote_prefix),
                "bucket": self.bucket,
                "raw_remote_prefix": str(self.raw_remote_prefix),
                "log_root": str(self.log_root),
                "data_h5_s3": str(hdf5),
                "out_dir": str(self.processed_local_root),
                "fps": 30,
                "chunk_timesteps": 100,
                "save_mp4": True,
            }
            name = hdf5.stem
            yield name, args

    @staticmethod
    def convert_one_bundle(
        processed_local_root: str,
        processed_remote_prefix: str,
        bucket: str,
        raw_remote_prefix: str,
        log_root: str,
        data_h5_s3: str,
        out_dir: str,
        arm: str,
        fps: int,
        task_name: str,
        task_description: str,
        chunk_timesteps: int,
        save_mp4: bool,
    ) -> tuple[str, str, int]:
        s3_client = get_boto3_s3_client()
        processed_local_root = Path(processed_local_root)
        processed_remote_prefix = processed_remote_prefix.rstrip("/")
        bucket = bucket.rstrip("/")
        log_root = Path(log_root)
        raw_remote_prefix = raw_remote_prefix.rstrip("/")
        hdf5_s3 = S3Path(data_h5_s3)
        stem = hdf5_s3.stem

        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{stem}-{uuid.uuid4().hex[:8]}.log"

        tmp_dir = Path.home() / "temp_eva_processing" / f"{stem}-{uuid.uuid4().hex[:6]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", encoding="utf-8") as log_fh:
            tee_out = _Tee(sys.stdout, log_fh)
            tee_err = _Tee(sys.stderr, log_fh)
            with (
                contextlib.redirect_stdout(tee_out),
                contextlib.redirect_stderr(tee_err),
            ):
                print(f"[LOG] {stem}: {log_path}", flush=True)

                raw_bucket, raw_prefix = _parse_s3_uri(
                    raw_remote_prefix, default_bucket=bucket
                )
                raw_root = S3Path(raw_remote_prefix)

                rel = hdf5_s3.relative_to(raw_root).as_posix()
                t_key = f"{raw_prefix.rstrip('/')}/{rel}".strip("/")
                local_hdf5 = tmp_dir / hdf5_s3.name
                try:
                    s3_client.download_file(raw_bucket, t_key, str(local_hdf5))
                except Exception as e:
                    print(
                        f"[ERR] aws download failed for {data_h5_s3}: {e}", flush=True
                    )
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return "", "", -1

                ds_parent = Path(out_dir)
                ds_parent.mkdir(parents=True, exist_ok=True)
                ds_path = ds_parent

                try:
                    print(
                        f"[INFO] Converting: {stem} → {ds_path} (arm={arm})",
                        flush=True,
                    )
                    job_kwargs = dict(
                        raw_path=str(local_hdf5),
                        output_dir=str(ds_parent),
                        arm=arm,
                        fps=fps,
                        task_name=task_name,
                        task_description=task_description,
                        chunk_timesteps=chunk_timesteps,
                        save_mp4=save_mp4,
                    )
                    zarr_path, mp4_path = EvaRay.zarr_job(**job_kwargs)
                    frames = -1
                    zarr_store_path = zarr_path
                    info = zarr_store_path / "zarr.json"
                    print(f"[DEBUG] Zarr metadata path: {info}", flush=True)
                    if info.exists():
                        try:
                            meta = json.loads(info.read_text())
                            print(
                                f"[DEBUG] Zarr metadata keys: {list(meta.keys())}",
                                flush=True,
                            )
                            frames = int(
                                meta.get("attributes", {}).get("total_frames", -1)
                            )
                        except Exception as e:
                            print(
                                f"[ERR] Failed to parse zarr metadata {info}: {e}",
                                flush=True,
                            )
                            frames = -1
                    else:
                        print(f"[ERR] Zarr metadata not found: {info}", flush=True)

                    try:
                        out_bucket, out_prefix = _parse_s3_uri(
                            processed_remote_prefix, default_bucket=bucket
                        )
                        zarr_filename = Path(zarr_path).stem
                        zarr_s3_key = (
                            f"{out_prefix.rstrip('/')}/{zarr_filename}.zarr".strip("/")
                        )
                        mp4_s3_key = None
                        if mp4_path:
                            mp4_s3_key = (
                                f"{out_prefix.rstrip('/')}/{Path(mp4_path).name}".strip(
                                    "/"
                                )
                            )
                        _cleanup_existing_processed_outputs(
                            bucket=out_bucket,
                            zarr_prefix=zarr_s3_key,
                            mp4_key=mp4_s3_key,
                        )
                        upload_dir_to_s3(
                            str(zarr_store_path), out_bucket, prefix=zarr_s3_key
                        )
                        shutil.rmtree(str(zarr_store_path), ignore_errors=True)
                        print(
                            f"[CLEANUP] Removed local zarr store: {zarr_store_path}",
                            flush=True,
                        )
                        if mp4_path:
                            s3_client.upload_file(str(mp4_path), out_bucket, mp4_s3_key)
                            Path(mp4_path).unlink(missing_ok=True)
                            print(
                                f"[CLEANUP] Removed local mp4: {mp4_path}", flush=True
                            )
                    except Exception as e:
                        print(
                            f"[ERR] Failed to upload {zarr_store_path} to S3: {e}",
                            flush=True,
                        )
                        return "", "", -2

                    return str(zarr_path), str(mp4_path), frames

                except Exception as e:
                    err_msg = f"[FAIL] {stem}: {e}\n{traceback.format_exc()}"
                    print(err_msg, flush=True)
                    return "", "", -1
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def zarr_job(
        raw_path: str | Path,
        output_dir: str | Path,
        arm: str,
        fps: int = 30,
        save_mp4: bool = True,
        task_name: str = "",
        task_description: str = "",
        chunk_timesteps: int = 100,
    ) -> tuple[Path, Path] | None:
        """
        Convert one <vrs, vrs.json, mps_*> trio to a Zarr dataset.
        """
        raw_path = Path(raw_path).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()

        args = SimpleNamespace(
            raw_path=raw_path,
            output_dir=output_dir,
            arm=arm,
            fps=fps,
            save_mp4=save_mp4,
            task_name=task_name,
            task_description=task_description,
            chunk_timesteps=chunk_timesteps,
        )

        return eva_main(args)
