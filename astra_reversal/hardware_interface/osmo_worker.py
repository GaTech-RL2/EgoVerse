"""Own-prefix OSMO commissioning and archival; no physical hardware or API calls."""

import json
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import file_hash, write_json
from .preflight import probe
from .protocol import load_manifest


def archive(root, workflow):
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    prefix = "experiments/libero-hardware-interface-20261008/" + workflow + "/"
    if client.list_objects_v2(Bucket="rldb", Prefix=prefix, MaxKeys=1).get("KeyCount"):
        raise FileExistsError("archive_prefix_already_exists")
    objects = []
    # Never archive a live scratch jail, environment, or credential file.
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path.is_symlink()
            or any("jail" in p or p == "scratch-runtime" for p in relative.parts)
        ):
            continue
        key = prefix + str(relative)
        client.upload_file(str(path), "rldb", key)
        objects.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
        )
    receipt = {"workflow": workflow, "prefix": prefix, "files": objects}
    client.put_object(
        Bucket="rldb",
        Key=prefix + "archive-receipt.json",
        Body=json.dumps(receipt).encode(),
    )
    print(json.dumps({"archive_prefix": prefix, "files": len(objects)}), flush=True)


def main():
    root = Path("artifacts")
    manifest = load_manifest(
        "experiments/libero_hardware_interface/preregistration.yaml"
    )
    write_json(
        root / "source-receipt.json",
        {
            "commit": os.environ["HARDWARE_SOURCE_COMMIT"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "container": manifest["container_digest"],
            "workflow": os.environ["HARDWARE_WORKFLOW"],
        },
    )
    try:
        resolved, checks = probe("upstream-libero", root / "prepared", manifest)
        suites = (
            ET.parse(root / "runtime/unit-tests.xml").getroot().findall("testsuite")
        )
        if not suites or any(
            int(s.get("failures", "0")) + int(s.get("errors", "0")) for s in suites
        ):
            raise RuntimeError("contract_tests_failed")
        checks["success_per_step"] = checks["budget_termination"] = True
        manifest["resolved"], manifest["readiness"] = resolved, checks
        import yaml

        (root / "prepared/preregistration.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False)
        )
        write_json(
            root / "commissioning.json",
            {
                "gates": checks,
                "resolved": resolved,
                "unit_tests_passed": sum(
                    int(s.get("tests", "0")) - int(s.get("skipped", "0"))
                    for s in suites
                ),
                "model_trials_started": 0,
                "scored_collection_allowed": False,
            },
        )
        print(json.dumps({"gates": checks, "model_trials_started": 0}), flush=True)
    except BaseException as error:
        write_json(
            root / "failure.json",
            {
                "error_class": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        (root / "bootstrap.log").write_bytes(Path("bootstrap.log").read_bytes())
        archive(root, os.environ["HARDWARE_WORKFLOW"])


if __name__ == "__main__":
    main()
