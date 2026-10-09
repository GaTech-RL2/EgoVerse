"""Prepare pinned RoboCasa inference assets on a GPU-zero OSMO worker."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(args, **kwargs):
    print(json.dumps({"command": args[:5], "time": time.time()}), flush=True)
    subprocess.run(args, check=True, **kwargs)


def clone(source, target):
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            source["url"],
            str(target),
        ],
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
    )
    run(
        ["git", "checkout", "--detach", source["revision"]],
        cwd=target,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
    )
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=target, text=True
    ).strip()
    if actual != source["revision"]:
        raise ValueError("Dependency source revision does not match the manifest")


def fetch_asset(item, base):
    target = base / item["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"asset_download_start": item["name"]}), flush=True)
    temporary = target.with_suffix(".partial")
    with (
        urllib.request.urlopen(item["url"], timeout=120) as response,
        temporary.open("wb") as out,
    ):
        total = 0
        while block := response.read(8 * 1024 * 1024):
            total += len(block)
            if total > 25 * 1024**3:
                raise ValueError(
                    "Asset archive exceeds the declared per-file storage limit"
                )
            out.write(block)
    temporary.replace(target)
    with zipfile.ZipFile(target) as archive:
        for info in archive.infolist():
            if Path(info.filename).is_absolute() or ".." in Path(info.filename).parts:
                raise ValueError("Unsafe upstream asset archive member")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError("Upstream asset archive failed its CRC check")
    record = {**item, "bytes": total, "sha256": sha256(target)}
    print(
        json.dumps({"asset_download_complete": item["name"], "bytes": total}),
        flush=True,
    )
    return record


def build_environment(root):
    runtime = root / "runtime"
    run([sys.executable, "-m", "venv", "--system-site-packages", str(runtime)])
    python = str(runtime / "bin/python")
    run(
        [
            sys.executable,
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "jax[cuda12]==0.5.3",
            "flax==0.10.2",
            "orbax-checkpoint==0.11.13",
            "tensorstore==0.1.74",
            "ml-dtypes==0.4.1",
            "numpy==2.2.5",
            "numba==0.61.2",
            "scipy==1.15.3",
            "mujoco==3.3.1",
            "augmax>=0.3.4",
            "dm-tree>=0.1.8",
            "chex==0.1.89",
            "tqdm-loggable==0.2",
            "einops>=0.8",
            "equinox>=0.11.8",
            "jaxtyping==0.2.36",
            "numpydantic>=1.6.6",
            "sentencepiece>=0.2",
            "beartype==0.19.0",
            "treescope>=0.1.7",
            "transformers==4.53.2",
            "ml_collections==1.0.0",
            "fsspec[gcs]",
            "opencv-python-headless<4.12",
            "imageio[ffmpeg]",
            "pillow",
            "pyyaml",
            "pynput",
            "pygame",
            "h5py",
            "lxml",
            "hidapi",
            "termcolor",
            "gymnasium",
            "qpsolvers[quadprog]>=4.3.1",
            "boto3",
            "huggingface_hub",
            "tyro>=0.9.5",
            "rich",
            "msgpack",
            "websockets",
            "etils[epath]",
        ]
    )
    for relative in (
        "robosuite",
        "robocasa",
        "openpi",
        "openpi/packages/openpi-client",
    ):
        run(
            [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                "-e",
                str(root / "sources" / relative),
            ]
        )
    run([python, "-m", "pip", "freeze"], stdout=(root / "runtime_freeze.txt").open("w"))


def archive_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", connect_timeout=30, read_timeout=180),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/opt/astra-complex"))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "sources").mkdir()
    workflow = os.environ.get("ASTRA_RUN_ID", "local-stage")
    if not re.fullmatch(r"astra-complex-20261006-stage-[0-9]+|local-stage", workflow):
        raise ValueError("Unexpected staging workflow identity")
    checkpoint = manifest["checkpoints"]["robocasa_pi05"]
    from huggingface_hub import snapshot_download

    def fetch_weights():
        downloaded = Path(
            snapshot_download(
                checkpoint["repo_id"],
                revision=checkpoint["revision"],
                allow_patterns=[f["path"] for f in checkpoint["files"]],
                local_dir=root / "weights",
                max_workers=4,
            )
        )
        records = []
        for item in checkpoint["files"]:
            path = downloaded / item["path"]
            actual = sha256(path)
            if path.stat().st_size != item["size"]:
                raise ValueError("Checkpoint file size differs from pinned metadata")
            expected = (item.get("lfs") or {}).get("sha256")
            if expected and actual != expected:
                raise ValueError("Checkpoint shard SHA-256 differs from pinned release")
            records.append(
                {"path": item["path"], "sha256": actual, "bytes": path.stat().st_size}
            )
        return records

    with ThreadPoolExecutor(max_workers=4) as pool:
        weights = pool.submit(fetch_weights)
        assets = [
            pool.submit(fetch_asset, item, root / "asset_archives")
            for item in manifest["robocasa_asset_archives"]
        ]
        for name in ("robosuite", "robocasa", "openpi"):
            clone(manifest["sources"][name], root / "sources" / name)
        build_environment(root)
        weight_records = weights.result()
        asset_records = [future.result() for future in assets]
    receipt = {
        "schema": "astra-complex-staging-1",
        "workflow": workflow,
        "manifest_sha256": sha256(args.manifest),
        "gpu_count": 0,
        "policy_rollouts": 0,
        "weights": weight_records,
        "assets": asset_records,
        "sources": {
            k: manifest["sources"][k] for k in ("robosuite", "robocasa", "openpi")
        },
    }
    # The immutable stage receipt records the input bytes. It does not certify inference.
    (root / "stage_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if args.upload:
        client = archive_client()
        prefix = f"experiments/astra-complex-20261006/{workflow}"
        if client.list_objects_v2(Bucket="rldb", Prefix=prefix + "/", MaxKeys=1).get(
            "KeyCount"
        ):
            raise FileExistsError("Refuse to overwrite an existing staging archive")
        # Store weights and original asset zips individually: avoid another local
        # 25+ GB archive and permit independent checksummed downloads on GPU workers.
        upload_records = []
        for category, records in (
            ("weights", weight_records),
            ("asset_archives", asset_records),
        ):
            for item in records:
                rel = item.get("path", item.get("name"))
                path = root / category / rel
                key = f"{prefix}/{category}/{rel}"
                client.upload_file(str(path), "rldb", key)
                upload_records.append(
                    {
                        "key": key,
                        "relative": f"{category}/{rel}",
                        "bytes": path.stat().st_size,
                        "sha256": item["sha256"],
                    }
                )
        bundle = root.parent / "astra-complex-runtime.tar.gz"
        with tarfile.open(bundle, "w:gz", compresslevel=1) as tar:
            for name in (
                "sources",
                "runtime",
                "runtime_freeze.txt",
                "stage_receipt.json",
            ):
                tar.add(
                    root / name,
                    arcname=name,
                    filter=lambda entry: None if "/.git/" in entry.name else entry,
                )
        key = f"{prefix}/runtime.tar.gz"
        client.upload_file(str(bundle), "rldb", key)
        upload_records.append(
            {
                "key": key,
                "relative": "runtime.tar.gz",
                "bytes": bundle.stat().st_size,
                "sha256": sha256(bundle),
            }
        )
        receipt["uploaded_files"] = upload_records
        receipt["status"] = "staged_not_evaluated"
        client.put_object(
            Bucket="rldb",
            Key=f"{prefix}/receipt.json",
            Body=json.dumps(receipt, indent=2).encode(),
            ContentType="application/json",
        )
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "receipt_key": f"{prefix}/receipt.json",
                    "files": len(upload_records),
                    "bytes": sum(r["bytes"] for r in upload_records),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
