"""Stage the two requested Bench2Dex releases without allocating a GPU."""

import argparse
import json
import os
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from stage_robocasa import archive_client, clone, sha256


def download(repository, root):
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository["repo_id"],
        repo_type=repository.get("repo_type", "model"),
        revision=repository["revision"],
        allow_patterns=[item["path"] for item in repository["files"]],
        local_dir=root,
        max_workers=8,
    )
    records = []
    for item in repository["files"]:
        path = root / item["path"]
        if path.stat().st_size != item["size"]:
            raise ValueError("A selected release file differs from pinned metadata")
        actual = sha256(path)
        expected = (item.get("lfs") or {}).get("sha256")
        if expected and actual != expected:
            raise ValueError("Release LFS digest mismatch")
        records.append(
            {"path": item["path"], "bytes": path.stat().st_size, "sha256": actual}
        )
    print(
        json.dumps(
            {
                "download_verified": repository["repo_id"],
                "files": len(records),
                "bytes": sum(r["bytes"] for r in records),
            }
        ),
        flush=True,
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/opt/astra-bench"))
    args = parser.parse_args()
    workflow = os.environ["ASTRA_RUN_ID"]
    if not re.fullmatch(r"astra-complex-20261006-bench-stage-[0-9]+", workflow):
        raise ValueError("Unexpected Bench2Dex staging identity")
    release = json.loads(args.release_manifest.read_text())
    assets = json.loads(args.asset_manifest.read_text())
    anchors = json.loads(args.anchor_manifest.read_text())
    checkpoints = [
        release["checkpoints"][name]
        for name in ("bench2dex_jigsaw_pi05", "bench2dex_fridge_pi05")
    ]
    if len({(x["repo_id"], x["revision"]) for x in checkpoints}) != 1:
        raise ValueError("Expected both models from the same pinned release")
    weights = {
        "repo_id": checkpoints[0]["repo_id"],
        "revision": checkpoints[0]["revision"],
        "files": [f for checkpoint in checkpoints for f in checkpoint["files"]],
    }
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    client = archive_client()
    prefix = f"experiments/astra-complex-20261006/{workflow}"
    if client.list_objects_v2(Bucket="rldb", Prefix=prefix + "/", MaxKeys=1).get(
        "KeyCount"
    ):
        raise FileExistsError("Refuse to overwrite a previous stage")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            name: pool.submit(download, repository, root / name)
            for name, repository in (
                ("policy_ckpt", weights),
                ("dex2bench_dataset", assets),
                ("teleopdata", anchors),
            )
        }
        clone(release["sources"]["bench2dex"], root / "Bench2Dex")
        records = {name: future.result() for name, future in futures.items()}
    receipt = {
        "workflow": workflow,
        "status": "assets_staged_simulator_and_policy_not_evaluated",
        "gpu_count": 0,
        "policy_rollouts": 0,
        "manifest_sha256": {
            name: sha256(path)
            for name, path in (
                ("release", args.release_manifest),
                ("assets", args.asset_manifest),
                ("anchors", args.anchor_manifest),
            )
        },
        "records": records,
        "sources": {"bench2dex": release["sources"]["bench2dex"]},
        "asset_scope": assets["selection"],
        "anchor_role": "Native reset/reference anchors only. Training/evaluation roles must be explicitly assigned before learning.",
    }
    (root / "stage_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    uploaded = []
    for item in records["policy_ckpt"]:
        path = root / "policy_ckpt" / item["path"]
        key = f"{prefix}/policy_ckpt/{item['path']}"
        client.upload_file(str(path), "rldb", key)
        uploaded.append(
            {
                "key": key,
                "relative": "policy_ckpt/" + item["path"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
        )
    archive_path = root.parent / "astra-bench-assets.tar.gz"

    def included(entry):
        return (
            None
            if any(p in {".git", ".cache"} for p in Path(entry.name).parts)
            else entry
        )

    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        for name in (
            "Bench2Dex",
            "dex2bench_dataset",
            "teleopdata",
            "stage_receipt.json",
        ):
            archive.add(root / name, arcname=name, filter=included)
    key = f"{prefix}/assets.tar.gz"
    client.upload_file(str(archive_path), "rldb", key)
    uploaded.append(
        {
            "key": key,
            "relative": "assets.tar.gz",
            "bytes": archive_path.stat().st_size,
            "sha256": sha256(archive_path),
        }
    )
    receipt["uploaded_files"] = uploaded
    client.put_object(
        Bucket="rldb",
        Key=f"{prefix}/receipt.json",
        Body=json.dumps(receipt, indent=2).encode(),
        ContentType="application/json",
    )
    print(
        json.dumps(
            {
                "workflow": workflow,
                "status": receipt["status"],
                "files": len(uploaded),
                "bytes": sum(item["bytes"] for item in uploaded),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
