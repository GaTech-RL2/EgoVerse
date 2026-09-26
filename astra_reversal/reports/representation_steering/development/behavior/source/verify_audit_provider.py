"""Mutation checks on a preserved live prefix; no experiment/provider calls."""

import argparse
import copy
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "provider_audit", HERE / "audit_provider.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--archive-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="representation-audit-regression-") as temp:
        directory = Path(temp)
        worker, task = directory / "worker", directory / "worker/task"
        task.mkdir(parents=True)
        for path in args.worker_dir.glob("*.json"):
            shutil.copy2(path, worker / path.name)
        for name in ("summary.json", "events.jsonl", "provider.jsonl"):
            shutil.copy2(args.task_dir / name, task / name)
        original_provider = (task / "provider.jsonl").read_bytes()
        original_events = (task / "events.jsonl").read_bytes()
        original = audit.audit_case(task, worker)
        cases = [{"name": "unchanged_prefix_passes", "passed": True}]
        provider_rows = [json.loads(row) for row in original_provider.splitlines()]
        event_rows = [json.loads(row) for row in original_events.splitlines()]
        assert provider_rows and any(
            row["kind"] == "representation_request" for row in event_rows
        )
        for mutation in (
            "wire_payload_hash",
            "duplicate_physical_call",
            "request_png_raw_binding",
            "cross_arm_feedback",
            "stale_applied_decision",
        ):
            providers, events = copy.deepcopy(provider_rows), copy.deepcopy(event_rows)
            if mutation == "wire_payload_hash":
                providers[0]["payload_sha256"] = "0" * 64
            elif mutation == "duplicate_physical_call":
                providers.append(copy.deepcopy(providers[0]))
            elif mutation == "request_png_raw_binding":
                target = next(
                    row
                    for row in events
                    if row["kind"] == "representation_generation"
                    and row["attempt_id"].startswith("astra_")
                    and row["observation_step"] == 0
                )
                target["observation"]["observation/image"]["sha256"] = "0" * 64
            elif mutation == "cross_arm_feedback":
                target = next(
                    row for row in events if row["kind"] == "representation_request"
                )
                target["request"]["completed_rollout_feedback"][0]["attempt_id"] = (
                    "another_arm_revision1"
                )
            else:
                target = next(
                    row
                    for row in events
                    if row["kind"] == "representation_generation"
                    and row["attempt_id"].startswith("astra_")
                    and row["active_intervention"] is not None
                )
                target["active_intervention"] = None
            (task / "provider.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in providers)
            )
            (task / "events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events)
            )
            try:
                audit.audit_case(task, worker)
            except ValueError as exc:
                cases.append({"name": mutation, "passed": True, "rejection": str(exc)})
            else:
                raise AssertionError(f"Mutation was not rejected: {mutation}")
        if args.archive_receipt:
            (task / "provider.jsonl").write_bytes(original_provider)
            (task / "events.jsonl").write_bytes(original_events)
            archive = json.loads(args.archive_receipt.read_text())
            audit.audit_case(task, worker, archive_receipt=args.archive_receipt)
            cases.append({"name": "actual_archive_binding_passes", "passed": True})
            changed = copy.deepcopy(archive)
            changed["input_file_sha256"]["events.jsonl"] = "0" * 64
            changed_path = directory / "wrong_archive.json"
            changed_path.write_text(json.dumps(changed))
            try:
                audit.audit_case(task, worker, archive_receipt=changed_path)
            except ValueError as exc:
                cases.append(
                    {
                        "name": "archive_event_hash_mismatch",
                        "passed": True,
                        "rejection": str(exc),
                    }
                )
            else:
                raise AssertionError("Mismatched archive bytes were not rejected")
            if archive["status"] == "verified_partial_archive":
                try:
                    audit.audit_case(
                        task,
                        worker,
                        archive_receipt=args.archive_receipt,
                        require_complete=True,
                    )
                except ValueError as exc:
                    cases.append(
                        {
                            "name": "partial_archive_cannot_claim_complete",
                            "passed": True,
                            "rejection": str(exc),
                        }
                    )
                else:
                    raise AssertionError("Partial archive was promoted to complete")
        result = {
            "schema_version": "representation-provider-auditor-mutation-validation-1.0",
            "status": "passed",
            "auditor_sha256": audit.file_sha(HERE / "audit_provider.py"),
            "checker_sha256": audit.file_sha(__file__),
            "codec": audit.codec(),
            "source_prefix": original["source_file_sha256"],
            "archive_receipt_sha256": (
                audit.file_sha(args.archive_receipt)
                if args.archive_receipt is not None
                else None
            ),
            "checks": cases,
            "new_provider_calls": 0,
            "scope": "Offline mutations of a private copy only; experiment files and recorded outcomes unchanged.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "checks": len(cases),
                    "sha256": audit.file_sha(args.output),
                }
            )
        )


if __name__ == "__main__":
    main()
