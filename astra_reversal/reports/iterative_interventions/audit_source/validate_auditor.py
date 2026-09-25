"""Offline tamper checks against the retained development records."""

import argparse
import copy
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from audit_worker import CaseAudit, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metadata = {
        name: json.loads((args.development / name).read_text())
        for name in (
            "runtime.json",
            "protocol.json",
            "frozen_plan.json",
            "reset_manifest.json",
            "checkpoint.json",
            "progress.json",
        )
    }
    data = {
        name: (args.development / "case_1_1" / name).read_bytes()
        for name in ("events.jsonl", "summary.json")
    }
    audit = CaseAudit("case_1_1", metadata, data, args.assets)
    probe = audit.initial["development_embedding_probe"]["conditioning"]
    raw_id = audit.initial["condition_id"]
    observation_id = audit.initial["zero_embedding_hook"]["observation_id"]
    checks = []

    def reject(label, function):
        try:
            function()
        except ValueError:
            checks.append({"check": label, "passed": True})
        else:
            raise AssertionError(label + " was not rejected")

    assert audit.conditioning(probe, raw_id, observation_id) == probe["condition_id"]
    checks.append({"check": "valid_weighted_provenance", "passed": True})
    for field, value in (
        ("operator_source_sha256", "0" * 64),
        ("vision_prefix_unchanged", False),
        ("original_token_mask_sha256", "0" * 64),
        ("guidance_token_sha256", "0" * 64),
        ("condition_id", "0" * 64),
        ("delta_frobenius", probe["bound_frobenius"] + 1),
    ):
        modified = copy.deepcopy(probe)
        modified[field] = value
        reject(
            "tampered_" + field,
            lambda modified=modified: audit.conditioning(
                modified, raw_id, observation_id
            ),
        )
    ref = audit.initial["known_noise"]
    with tarfile.open(args.development / "final_archive.tar.gz", "r:gz") as archive:
        original = archive.extractfile("results/case_1_1/" + ref["array"]).read()
    changed = np.load(io.BytesIO(original), allow_pickle=False)
    changed[0, 0, 0] += 1
    buffer = io.BytesIO()
    np.save(buffer, changed, allow_pickle=False)
    reject(
        "tampered_recorded_noise_array",
        lambda: audit.array(ref["array"], buffer.getvalue()),
    )
    audit.array(ref["array"], original)
    checks.append({"check": "valid_recorded_noise_array", "passed": True})
    reject("duplicate_array_member", lambda: audit.array(ref["array"], original))
    ids = [candidate_id for _, candidate_id in audit.proposals]
    assert len(ids) > len(set(ids))
    assert len(audit.proposals) == sum(
        row["kind"] == "intervention_proposal" for row in audit.rows
    )
    checks.append(
        {"check": "same_candidate_id_in_distinct_arms_retained", "passed": True}
    )
    write_json(
        args.output,
        {
            "status": "passed",
            "checks": checks,
            "audit_source_sha256": sha(
                Path(__file__).with_name("audit_worker.py").read_bytes()
            ),
            "validation_source_sha256": sha(Path(__file__).read_bytes()),
            "input_events_sha256": sha(data["events.jsonl"]),
            "no_model_api_or_simulator_calls": True,
        },
    )
    print(json.dumps({"status": "passed", "checks": len(checks)}))


if __name__ == "__main__":
    main()
