"""Run the preregistered pilot only after matching completed smoke receipts."""

import json
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from astra_reversal.hardware_interface.analysis import audit  # noqa: E402
from astra_reversal.hardware_interface.common import (  # noqa: E402
    strict_json,
    write_json,
)
from astra_reversal.hardware_interface.launcher import load_outcomes, run  # noqa: E402
from astra_reversal.hardware_interface.osmo_worker import archive  # noqa: E402
from astra_reversal.hardware_interface.preflight import probe, source_hash  # noqa: E402
from astra_reversal.hardware_interface.protocol import (  # noqa: E402
    load_manifest,
    save_schedule,
    schedule,
    validate,
)
from astra_reversal.hardware_interface.reporting import (  # noqa: E402
    export,
    power_worksheet,
)


def validate_commissioning(locks):
    manifest = load_manifest(locks / "ready-preregistration.yaml")
    validate(manifest, scored=True)
    smoke = strict_json((locks / "model-smoke.json").read_bytes())
    transport = strict_json((locks / "transport.json").read_bytes())
    if (
        smoke["result"]["trials"] != 3
        or smoke["result"]["split"] != "smoke"
        or smoke["audit"]["status"] != "passed"
        or {r["trial"] for r in smoke["audit"]["trials"]}
        != {"smoke-" + c for c in manifest["conditions"]}
        or any(r["status"] != "passed" for r in smoke["audit"]["trials"])
        or not transport["passed"]
        or transport["model"] != manifest["model"]
        or transport.get("arm_tool_schemas")
        != {c: True for c in manifest["conditions"]}
        or manifest["resolved"]["adapter_sha256"] != source_hash()
    ):
        raise ValueError("pilot_requires_matching_complete_live_commissioning")
    return manifest


def main():
    root, locks = Path("artifacts"), Path("pilot-locks")
    manifest, args = None, None
    try:
        manifest = validate_commissioning(locks)
        resolved, gates = probe("upstream-libero", root / "prepared", manifest)
        if resolved != manifest["resolved"]:
            raise ValueError("pilot_runtime_differs_from_live_smoke")
        suites = (
            ET.parse(root / "runtime/unit-tests.xml").getroot().findall("testsuite")
        )
        if not suites or any(
            int(s.get("failures", "0")) + int(s.get("errors", "0")) for s in suites
        ):
            raise RuntimeError("pilot_contract_tests_failed")
        gates["success_per_step"] = gates["budget_termination"] = True
        for condition in manifest["conditions"]:
            gates["model_smoke_" + condition] = True
        manifest["readiness"] = gates
        validate(manifest, scored=True)
        manifest_path = root / "pilot-preregistration.yaml"
        with manifest_path.open("x") as stream:
            stream.write(yaml.safe_dump(manifest, sort_keys=False))
        rows = schedule(
            manifest, strict_json((root / "prepared/catalog.json").read_bytes())
        )
        save_schedule(root / "trials.csv", rows)
        write_json(
            root / "source-receipt.json",
            {
                "commit": os.environ["HARDWARE_SOURCE_COMMIT"],
                "payload_sha256": os.environ["PAYLOAD_SHA256"],
                "workflow": os.environ["HARDWARE_WORKFLOW"],
                "live_smoke_source": strict_json(
                    (locks / "source-receipt.json").read_bytes()
                ),
                "adapter_sha256": source_hash(),
                "resolved": resolved,
            },
        )
        args = SimpleNamespace(
            manifest=str(manifest_path),
            prepared=str(root / "prepared"),
            libero_root="upstream-libero",
            out=str(root / "pilot"),
            split="pilot",
            schedule=str(root / "trials.csv"),
        )
        print(
            json.dumps(
                {
                    "pilot_start": True,
                    "trials": sum(r["split"] == "pilot" for r in rows),
                }
            ),
            flush=True,
        )
        result = run(args, manifest)
        print(json.dumps({"pilot_result": result}), flush=True)
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
        try:
            if args and (Path(args.out) / "runs").exists():
                outcomes = load_outcomes(Path(args.out) / "runs")
                write_json(root / "pilot-audit.json", audit(Path(args.out) / "runs"))
                export(outcomes, manifest, root / "results", split="pilot")
                expected = (
                    len(manifest["task_ids"])
                    * len(manifest["pilot_indices"])
                    * len(manifest["conditions"])
                    * manifest["replicates"]
                )
                if len(outcomes) == expected:
                    write_json(
                        root / "power-worksheet.json",
                        power_worksheet(outcomes, manifest),
                    )
        finally:
            (root / "bootstrap.log").write_bytes(Path("bootstrap.log").read_bytes())
            archive(root, os.environ["HARDWARE_WORKFLOW"])


if __name__ == "__main__":
    main()
