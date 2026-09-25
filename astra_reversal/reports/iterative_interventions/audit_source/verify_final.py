"""Verify frozen audit outputs and reset bindings without downloads or solves."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from frozen_intervention.records import digest


def require(value, message):
    if not value:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_snapshot(source):
    manifest = json.loads((source / "snapshot_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        require(file_hash(source / name) == expected, "Audit snapshot hash mismatch")
    return file_hash(source / "snapshot_manifest.json")


def array_identity(observation):
    return {
        name: {key: ref[key] for key in ("shape", "dtype", "sha256")}
        for name, ref in observation.items()
    }


def verify_reset_records(rows):
    """Bind every candidate reset to one manifest-bound case and raw observation."""
    cases = [row for row in rows if row["kind"] == "case"]
    initials = [row for row in rows if row["kind"] == "inversion_initialization"]
    require(len(cases) == len(initials) == 1, "Ambiguous case or initialization")
    entry, initial = cases[0]["entry"], initials[0]
    require(
        digest(np.asarray(entry["reset_state"], dtype=np.float64))
        == entry["reset_state_sha256"],
        "Reset state bytes differ from entry hash",
    )
    require(
        digest(
            {
                name: np.asarray(value, dtype=np.float64)
                for name, value in entry["reset_model"].items()
            }
        )
        == entry["reset_model_sha256"],
        "Reset model bytes differ from entry hash",
    )
    for name in ("known_noise", "recovered_noise"):
        require(
            initial[name]["sha256"] == initial[name + "_sha256"],
            "Initialization noise hash differs from saved array binding",
        )
    attempts = [row["attempt"] for row in rows if row["kind"] == "attempt"]
    require(bool(attempts), "No recorded rollout reset")
    first = attempts[0]["reset_audit"]
    for attempt in attempts:
        reset = dict(attempt["reset_audit"])
        recorded = reset.pop("sha256")
        require(digest(reset) == recorded, "Reset audit digest mismatch")
        require(attempt["reset_audit"] == first, "Candidate reset identity changed")
        for name in (
            "episode_id",
            "seed",
            "reset_state_sha256",
            "reset_model_sha256",
            "bddl_sha256",
        ):
            require(reset[name] == entry[name], "Reset differs from manifest entry")
        require(
            reset["post_stabilization_model_sha256"] == entry["reset_model_sha256"]
            and reset["post_stabilization_observation_sha256"]
            == initial["zero_embedding_hook"]["observation_id"],
            "Reset does not match checked initial condition",
        )
        require(
            reset["stabilization_steps"] == 10
            and reset["initial_success"] is False
            and reset["initial_terminated"] is False,
            "Invalid initial rollout state",
        )
    starts = [
        row
        for row in rows
        if row["kind"] == "candidate_generation" and row["observation_step"] == 0
    ]
    require(len(starts) == len(attempts), "Initial observation coverage mismatch")
    expected = array_identity(initial["observation"])
    require(
        all(array_identity(row["observation"]) == expected for row in starts),
        "Rollout initial arrays differ from common raw observation",
    )
    return {
        "episode_id": entry["episode_id"],
        "rollout_resets": len(attempts),
        "initial_observations": len(starts),
        "reset_audit_sha256": first["sha256"],
        "manifest_entry_sha256": digest(entry),
    }


def verify_final(root, source):
    snapshot_hash = verify_snapshot(source)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    require(
        summary["status"] == "complete_verified"
        and summary["archives"] == 8
        and summary["cases"] == 20
        and summary["exact_episode_coverage"]
        and summary["all_recording_checks_passed"]
        and not summary["failures"],
        "Final archive coverage is incomplete or invalid",
    )
    cases, workers = [], []
    for worker in range(8):
        directory = root / "evaluation" / f"worker_{worker}"
        report_path = directory / "audit.json"
        report = json.loads(report_path.read_text())
        require(
            file_hash(report_path) == summary["workers"][str(worker)]["report_sha256"]
            and report["source_sha256"] == file_hash(source / "audit_worker.py")
            and report["source_inventory_sha256"]
            == file_hash(source / "frozen_source_inventory.json")
            and report["auditor_wrapper_source_sha256"]
            == file_hash(source / "audit_worker_linux.py")
            and report["linux_vision_source_sha256"]
            == file_hash(source / "linux_vision.py"),
            "Worker audit source or result hash mismatch",
        )
        archive = report["archive"]
        require(
            archive["gzip_verified"]
            and archive["receipt_before"] == archive["receipt_after"]
            and archive["bytes"]
            == int(archive["receipt_after"]["content_range"].split("/")[-1]),
            "Incomplete immutable archive receipt",
        )
        case_hashes = {}
        for case in report["cases"]:
            case_dir = directory / case["case"]
            require(
                json.loads((case_dir / "audit.json").read_text()) == case,
                "Worker and case audit differ",
            )
            for name, key in (
                ("events.jsonl", "event_sha256"),
                ("summary.json", "summary_sha256"),
                ("array_inventory.json", "array_inventory_sha256"),
            ):
                require(
                    file_hash(case_dir / name) == case[key],
                    "Preserved case record hash mismatch",
                )
            rows = [
                json.loads(line)
                for line in (case_dir / "events.jsonl").read_text().splitlines()
                if line
            ]
            cases.append(verify_reset_records(rows))
            case_hashes[case["episode_id"]] = file_hash(case_dir / "audit.json")
        workers.append(
            {
                "worker": worker,
                "audit_sha256": file_hash(report_path),
                "archive": archive,
                "case_audits_sha256": case_hashes,
            }
        )
    ids = [case["episode_id"] for case in cases]
    expected = {
        f"{suite}:seed19:task{task}:state0"
        for suite in ("libero_goal_ood", "libero_spatial_ood")
        for task in range(10)
    }
    require(len(ids) == 20 and set(ids) == expected, "Wrong final episode coverage")
    return {
        "status": "passed",
        "summary_sha256": file_hash(summary_path),
        "snapshot_manifest_sha256": snapshot_hash,
        "validation_source_sha256": file_hash(Path(__file__)),
        "workers": workers,
        "cases": cases,
        "exact_episode_coverage": True,
        "rollout_resets_verified": sum(case["rollout_resets"] for case in cases),
        "no_new_model_api_simulator_or_network_calls": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_final(args.audit_root, Path(__file__).parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(result["cases"])}))


if __name__ == "__main__":
    main()
