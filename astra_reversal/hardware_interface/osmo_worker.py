"""Own-prefix OSMO commissioning, optional live model smoke, and archival."""

import copy
import json
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from .common import file_hash, write_json
from .preflight import probe
from .protocol import RESOURCE_CAPS, load_manifest, save_schedule, schedule, validate


def collect_pilot(root, manifest):
    """A fresh cohort after same-source native, transport and three-arm checks."""
    import yaml

    from .analysis import audit
    from .common import strict_json
    from .launcher import load_outcomes, run
    from .reporting import export, power_worksheet

    validate(manifest, scored=True)
    if any(manifest["limits"][key] is not None for key in RESOURCE_CAPS):
        raise ValueError("revised_pilot_requires_no_resource_budgets")
    path = root / "pilot-preregistration.yaml"
    with path.open("x") as stream:
        stream.write(yaml.safe_dump(manifest, sort_keys=False))
    rows = schedule(
        manifest, strict_json((root / "prepared/catalog.json").read_bytes())
    )
    save_schedule(root / "trials.csv", rows)
    args = SimpleNamespace(
        manifest=str(path),
        prepared=str(root / "prepared"),
        libero_root="upstream-libero",
        out=str(root / "pilot"),
        split="pilot",
        schedule=str(root / "trials.csv"),
    )
    expected = sum(row["split"] == "pilot" for row in rows)
    print(
        json.dumps({"pilot_start": True, "trials": expected, "resource_budgets": None}),
        flush=True,
    )
    try:
        result = run(args, manifest)
        print(json.dumps({"pilot_result": result}), flush=True)
    finally:
        if (root / "pilot/runs").exists():
            outcomes = load_outcomes(root / "pilot/runs")
            write_json(root / "pilot-audit.json", audit(root / "pilot/runs"))
            export(outcomes, manifest, root / "results", split="pilot")
            if len(outcomes) == expected:
                write_json(
                    root / "power-worksheet.json", power_worksheet(outcomes, manifest)
                )


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
        stage = os.environ.get("HARDWARE_STAGE", "commission")
        if stage in ("smoke", "pilot"):
            from .analysis import audit
            from .launcher import run
            from .model_smoke import transport_smoke

            transport_smoke(manifest, root / "prepared", root / "model-transport")
            smoke_manifest = copy.deepcopy(manifest)
            smoke_path = root / "prepared/preregistration.yaml"
            if stage == "pilot":
                # This is a short execution/replay check, never a task score.
                # The actual pilot retains its full 1,000-step horizon.
                smoke_manifest["limits"]["steps"] = 20
                smoke_path = root / "smoke-preregistration.yaml"
                with smoke_path.open("x") as stream:
                    stream.write(yaml.safe_dump(smoke_manifest, sort_keys=False))
            args = SimpleNamespace(
                manifest=str(smoke_path),
                prepared=str(root / "prepared"),
                libero_root="upstream-libero",
                out=str(root / "smoke"),
                split="pilot",
            )
            result = run(args, smoke_manifest, smoke=True)
            audits = audit(root / "smoke/runs")
            if len(audits["trials"]) != 3 or any(
                trial["status"] != "passed" for trial in audits["trials"]
            ):
                raise RuntimeError("model_smoke_replay_audit_failed")
            for condition in manifest["conditions"]:
                checks["model_smoke_" + condition] = True
            write_json(
                root / "model-smoke.json",
                {
                    "result": result,
                    "audit": audits,
                    "gates": checks,
                    "scored_results": False,
                    "commissioning_horizon_steps": smoke_manifest["limits"]["steps"],
                    "pilot_horizon_steps": manifest["limits"]["steps"],
                },
            )
            # Preserve the manifest used for replay. Readiness is a new artifact.
            (root / "ready-preregistration.yaml").write_text(
                yaml.safe_dump(manifest, sort_keys=False)
            )
            print(json.dumps({"model_smoke": result, "gates": checks}), flush=True)
            if stage == "pilot":
                collect_pilot(root, manifest)
        elif stage != "commission":
            raise ValueError("unknown_hardware_stage")
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
