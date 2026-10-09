"""Preserve bootstrap diagnostics even if benchmark dependencies failed to install."""

import hashlib
import json
import os
from pathlib import Path

import boto3

workflow = os.environ["HARDWARE_WORKFLOW"]
if not workflow.startswith("robocasa-hardware-interface-20261009-"):
    raise ValueError("archive_workflow_binding")
prefix = (
    f"experiments/robocasa-hardware-interface-20261009/{workflow}/bootstrap-failure/"
)
client = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
)
if client.list_objects_v2(Bucket="rldb", Prefix=prefix, MaxKeys=1).get("KeyCount"):
    raise FileExistsError("archive_prefix_already_exists")
root = Path("artifacts/runtime")
files = []
for path in sorted(root.glob("*")):
    if not path.is_file() or path.is_symlink():
        continue
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    client.upload_file(str(path), "rldb", prefix + path.name)
    files.append(
        {"path": path.name, "sha256": value.hexdigest(), "bytes": path.stat().st_size}
    )
client.put_object(
    Bucket="rldb",
    Key=prefix + "archive-receipt.json",
    Body=json.dumps({"workflow": workflow, "files": files}).encode(),
)
print(json.dumps({"bootstrap_diagnostics_archived": prefix, "files": len(files)}))
