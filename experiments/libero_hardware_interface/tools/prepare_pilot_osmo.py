"""Package an OSMO pilot with completed, source-matched model smoke receipts."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import yaml
from pilot_worker import validate_commissioning
from prepare_osmo import prepare


def prepare_pilot(destination, commissioning):
    commissioning = Path(commissioning).resolve()
    inputs = {
        "ready-preregistration.yaml": commissioning / "ready-preregistration.yaml",
        "model-smoke.json": commissioning / "model-smoke.json",
        "transport.json": commissioning / "model-transport/transport.json",
        "source-receipt.json": commissioning / "source-receipt.json",
    }
    # Validate before preparing a workflow. No unfinished smoke can enable a pilot.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="hardware-pilot-locks-") as temporary:
        locks = Path(temporary)
        for name, source in inputs.items():
            (locks / name).write_bytes(source.read_bytes())
        manifest = validate_commissioning(locks)
    receipt = prepare(destination, stage="smoke")
    destination = Path(destination).resolve()
    payload = destination / "payload.tar.gz"
    replacement = destination / "pilot-payload.tar.gz"
    lock_hashes = {}
    with (
        tarfile.open(payload, "r:gz") as existing,
        tarfile.open(replacement, "w:gz") as combined,
    ):
        for member in existing.getmembers():
            combined.addfile(member, existing.extractfile(member))
        for name, source in inputs.items():
            raw = source.read_bytes()
            member = tarfile.TarInfo("pilot-locks/" + name)
            member.size = len(raw)
            member.mode = 0o600
            combined.addfile(member, io.BytesIO(raw))
            lock_hashes[name] = hashlib.sha256(raw).hexdigest()
    replacement.replace(payload)
    sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    bootstrap = destination / "bootstrap.sh"
    before = "python -m astra_reversal.hardware_interface.osmo_worker"
    after = "python experiments/libero_hardware_interface/tools/pilot_worker.py"
    source = bootstrap.read_text()
    if source.count(before) != 1:
        raise ValueError("bootstrap_entrypoint_changed")
    bootstrap.write_text(source.replace(before, after))
    workflow = yaml.safe_load((destination / "workflow.yaml").read_text())
    workflow["workflow"]["name"] = "libero-hardware-interface-20261008-pilot"
    workflow["workflow"]["timeout"]["exec_timeout"] = "72h"
    workflow["workflow"]["tasks"][0]["environment"]["PAYLOAD_SHA256"] = sha
    (destination / "workflow.yaml").write_text(
        yaml.safe_dump(workflow, sort_keys=False)
    )
    receipt.update(
        {
            "scope": "150-trial preregistered unscored pilot; no confirmation launch",
            "stage": "pilot",
            "payload_sha256": sha,
            "bootstrap_sha256": hashlib.sha256(bootstrap.read_bytes()).hexdigest(),
            "commissioning_locks": lock_hashes,
            "adapter_sha256": manifest["resolved"]["adapter_sha256"],
        }
    )
    (destination / "launch-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument("--commissioning", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_pilot(args.destination, args.commissioning)))
