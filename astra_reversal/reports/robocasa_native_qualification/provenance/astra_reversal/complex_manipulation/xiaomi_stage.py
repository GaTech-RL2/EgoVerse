"""Stage the pinned, released Xiaomi RoboCasa policy without allocating a GPU."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from astra_reversal.complex_manipulation.stage_robocasa import (
    archive_client, clone, run, sha256,
)


def validate_release_file(path, item):
    """Verify both LFS payloads and ordinary Git blobs against the pinned release."""
    path = Path(path)
    if path.stat().st_size != item["size"]:
        raise ValueError("Release file size mismatch")
    if item.get("lfs"):
        if sha256(path) != item["lfs"]["sha256"]:
            raise ValueError("Release LFS SHA-256 mismatch")
    else:
        value = hashlib.sha1(f"blob {item['size']}\0".encode() + path.read_bytes())
        if value.hexdigest() != item["blobId"]:
            raise ValueError("Release Git blob hash mismatch")


def build_runtime(root):
    runtime = root / "runtime"
    run([sys.executable, "-m", "venv", str(runtime)])
    python = str(runtime / "bin/python")
    installer = [sys.executable, "-m", "uv", "pip", "install", "--python", python]
    run(installer + ["torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0",
                     "--index-url", "https://download.pytorch.org/whl/cu128"])
    run(installer + [
        "transformers==4.57.1", "numpy==2.2.5", "numba==0.61.2", "scipy==1.15.3",
        "mujoco==3.3.1", "opencv-python-headless<4.12", "imageio[ffmpeg]", "pillow",
        "pyyaml", "pynput", "pygame", "h5py", "lxml", "hidapi", "termcolor",
        "gymnasium", "qpsolvers[quadprog]>=4.3.1", "einops", "boto3", "setuptools",
    ])
    # The exact prebuilt wheel is the alternative recommended in DEPLOYMENT.md.
    run(installer + ["--no-deps", "https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"])
    for name in ("robosuite", "robocasa"):
        run(installer + ["--no-deps", "-e", str(root / "sources" / name)])
    with (root / "runtime_freeze.txt").open("w") as output:
        run([sys.executable, "-m", "uv", "pip", "freeze", "--python", python], stdout=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    workflow = os.environ["ASTRA_RUN_ID"]
    if not workflow.startswith("astra-complex-20261006-robocasa-xiaomi-stage-"):
        raise ValueError("Unexpected stage identity")
    root = Path("/opt/astra-xiaomi")
    root.mkdir(parents=True, exist_ok=False)
    (root / "sources").mkdir()
    client = archive_client()
    prefix = f"experiments/astra-complex-20261006/{workflow}"
    if client.list_objects_v2(Bucket="rldb", Prefix=prefix + "/", MaxKeys=1).get("KeyCount"):
        raise FileExistsError("Stage archive already exists")

    def weights():
        from huggingface_hub import snapshot_download
        checkpoint = manifest["checkpoint"]
        snapshot_download(checkpoint["repo_id"], revision=checkpoint["revision"],
                          local_dir=root / "weights", max_workers=4,
                          allow_patterns=[r["rfilename"] for r in checkpoint["files"]])
        records = []
        for item in checkpoint["files"]:
            path = root / "weights" / item["rfilename"]
            validate_release_file(path, item)
            records.append({"relative": "weights/" + item["rfilename"],
                            "bytes": path.stat().st_size, "sha256": sha256(path)})
        return records

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(weights)
        for name, source in manifest["sources"].items():
            clone(source, root / "sources" / name)
        build_runtime(root)
        records = future.result()
    # Check the real release processor on CPU before paying for GPU inference.
    script = ("from transformers import AutoProcessor; import robocasa, torch; "
              "p=AutoProcessor.from_pretrained('/opt/astra-xiaomi/weights',trust_remote_code=True); "
              "assert 'robocasa365' in p.list_robot_types(); "
              "print({'torch':torch.__version__,'robocasa':robocasa.__version__,"
              "'robot_types':p.list_robot_types()})")
    run([str(root / "runtime/bin/python"), "-c", script])
    old = json.loads(client.get_object(Bucket="rldb", Key=manifest["asset_receipt_key"])["Body"].read())
    if old["manifest_sha256"] != manifest["asset_manifest_sha256"]:
        raise ValueError("Existing assets do not match the original pinned simulator release")
    assets = [r for r in old["uploaded_files"] if r["relative"].startswith("asset_archives/")]
    if {Path(r["relative"]).name for r in assets} != {r["name"] for r in manifest["asset_archives"]}:
        raise ValueError("Missing original simulator asset archives")
    receipt = {"schema": "astra-xiaomi-stage-1", "workflow": workflow, "gpu_count": 0,
               "policy_rollouts": 0, "manifest_sha256": sha256(args.manifest),
               "source_revision": os.environ["ASTRA_SOURCE_REVISION"], "asset_files": assets,
               "sources": manifest["sources"], "checkpoint": manifest["checkpoint"]["revision"]}
    (root / "stage_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    bundle = root.parent / "astra-xiaomi-runtime.tar.gz"
    with tarfile.open(bundle, "w:gz", compresslevel=1) as tar:
        for name in ("sources", "runtime", "runtime_freeze.txt", "stage_receipt.json"):
            tar.add(root / name, arcname=name,
                    filter=lambda x: None if ".git" in Path(x.name).parts or
                    "__pycache__" in Path(x.name).parts else x)
    records.append({"relative": "runtime.tar.gz", "bytes": bundle.stat().st_size,
                    "sha256": sha256(bundle)})
    for row in records:
        path = bundle if row["relative"] == "runtime.tar.gz" else root / row["relative"]
        row["key"] = prefix + "/" + row["relative"]
        client.upload_file(str(path), "rldb", row["key"])
        print(json.dumps({"uploaded": row["relative"], "bytes": row["bytes"]}), flush=True)
    receipt.update(uploaded_files=records, status="staged_not_evaluated")
    client.put_object(Bucket="rldb", Key=prefix + "/receipt.json",
                      Body=json.dumps(receipt, indent=2).encode(), ContentType="application/json")
    print(json.dumps({"status": receipt["status"], "receipt_key": prefix + "/receipt.json"}), flush=True)


if __name__ == "__main__":
    main()
