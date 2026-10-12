"""Archive only this MAX commissioning task's own artifacts, without overwrite."""

import json
import os
from pathlib import Path

import boto3

from astra_reversal.hardware_interface.common import file_hash


def main():
    workflow = os.environ["HARDWARE_WORKFLOW"]
    source = os.environ["MAX_SOURCE"]
    if not workflow.startswith(
        "liberomax-hardware-interface-20261012-"
    ) or source not in ("plus", "pro"):
        raise ValueError("archive_workflow_binding")
    prefix = f"experiments/liberomax-hardware-interface-20261012/{workflow}/{source}/"
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    if client.list_objects_v2(Bucket="rldb", Prefix=prefix, MaxKeys=1).get("KeyCount"):
        raise FileExistsError("archive_prefix_already_exists")
    files = []
    for path in sorted(Path("artifacts").rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = str(path.relative_to("artifacts"))
        if "api-key" in relative or "jail" in relative:
            raise ValueError("private_artifact_not_allowed")
        client.upload_file(str(path), "rldb", prefix + relative)
        files.append(
            {"path": relative, "sha256": file_hash(path), "bytes": path.stat().st_size}
        )
    client.put_object(
        Bucket="rldb",
        Key=prefix + "archive-receipt.json",
        Body=json.dumps(
            {"workflow": workflow, "source": source, "files": files}
        ).encode(),
    )
    print(
        json.dumps({"archive_prefix": prefix, "files": len(files), "scored": False}),
        flush=True,
    )


if __name__ == "__main__":
    main()
