"""Bounded read-only polling for eight finalized evaluation archives."""

import argparse
import json
import time
from pathlib import Path

from audit_worker_linux import (
    BACKEND,
    Catalog,
    audit_worker,
    renderer_source_hashes,
    sha,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--live", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--duration", type=int, default=14400)
    args = parser.parse_args()
    deadline = time.monotonic() + args.duration
    completed, failures, last_gates = {}, {}, None
    while time.monotonic() < deadline:
        gates = []
        for path in sorted(args.live.glob("worker_*/case_*/summary.json")):
            try:
                data = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            initial = data.get("initialization")
            if initial:
                gates.append(
                    {
                        "episode_id": data["episode_id"],
                        "worker": int(path.parts[-3].split("_")[-1]),
                        "passed": initial["passed"],
                        "errors": initial["errors"],
                        "summary_prefix_sha256": sha(path.read_bytes()),
                    }
                )
        gate_key = [(row["episode_id"], row["passed"]) for row in gates]
        write_json(
            args.output / "live_numerical_gates.json",
            {"scope": "provisional_live_summaries", "cases": gates},
        )
        if gate_key != last_gates:
            print(
                json.dumps(
                    {
                        "event": "numerical_gates",
                        "observed": len(gates),
                        "passed": sum(row["passed"] for row in gates),
                    }
                ),
                flush=True,
            )
            last_gates = gate_key
        catalog = Catalog(args.catalog)
        for worker in range(8):
            output = args.output / "evaluation" / f"worker_{worker}"
            report = output / "audit.json"
            if report.exists():
                value = json.loads(report.read_text())
                if (
                    value["status"] == "complete_verified"
                    and value.get("vision_redraw_backend") == BACKEND
                    and all(
                        value.get(key) == expected
                        for key, expected in renderer_source_hashes().items()
                    )
                ):
                    completed[worker] = {
                        "report_sha256": sha(report.read_bytes()),
                        **value,
                    }
                    continue
            if worker in failures:
                continue
            try:
                catalog.receipt(f"worker_{worker}/artifacts.tar.gz")
            except Exception as exc:
                if getattr(exc, "code", None) not in (404, None):
                    print(
                        json.dumps(
                            {
                                "event": "archive_unavailable",
                                "worker": worker,
                                "exception": type(exc).__name__,
                                "http_status": getattr(exc, "code", None),
                            }
                        ),
                        flush=True,
                    )
                continue
            print(json.dumps({"event": "audit_start", "worker": worker}), flush=True)
            try:
                value = audit_worker(catalog, worker, output, args.assets)
            except Exception as exc:
                failure = {
                    "exception": type(exc).__name__,
                    "http_status": getattr(exc, "code", None),
                    "check": str(exc) if type(exc) is ValueError else None,
                }
                write_json(output / "audit_failure.json", failure)
                failures[worker] = failure
                print(
                    json.dumps({"event": "audit_failed", "worker": worker, **failure}),
                    flush=True,
                )
            else:
                completed[worker] = {"report_sha256": sha(report.read_bytes()), **value}
                print(
                    json.dumps(
                        {
                            "event": "audit_complete",
                            "worker": worker,
                            "cases": len(value["cases"]),
                            "archive_sha256": value["archive"]["sha256"],
                        }
                    ),
                    flush=True,
                )
        cases = [case for worker in completed.values() for case in worker["cases"]]
        ids = [case["episode_id"] for case in cases]
        expected = {
            f"{suite}:seed19:task{task}:state0"
            for suite in ("libero_goal_ood", "libero_spatial_ood")
            for task in range(10)
        }
        complete = len(completed) == 8 and len(ids) == 20 and set(ids) == expected
        summary = {
            "status": "complete_verified"
            if complete
            else "audit_issue"
            if failures
            else "partial_finalized_archives",
            "archives": len(completed),
            "cases": len(cases),
            "expected_cases": 20,
            "exact_episode_coverage": complete,
            "all_recording_checks_passed": not failures,
            "arrays_verified": sum(case["arrays_verified"] for case in cases),
            "generations_verified": sum(case["generations_verified"] for case in cases),
            "rollouts_verified": sum(case["rollouts"] for case in cases),
            "provider_calls": sum(case["provider"]["calls"] for case in cases),
            "provider_accepted": sum(case["provider"]["accepted"] for case in cases),
            "text_effect_generations": sum(
                case["candidate_text"].get("effect", 0) for case in cases
            ),
            "vision_changed_generations": sum(
                case["vision_changed_generations"] for case in cases
            ),
            "workers": {
                str(worker): {
                    key: value[key]
                    for key in (
                        "report_sha256",
                        "archive",
                        "source_sha256",
                        "source_inventory_sha256",
                        "vision_redraw_backend",
                        "auditor_wrapper_source_sha256",
                        "linux_vision_source_sha256",
                    )
                }
                for worker, value in completed.items()
            },
            "failures": failures,
            "no_new_model_or_api_calls": True,
        }
        write_json(args.output / "summary.json", summary)
        if complete:
            print(
                json.dumps(
                    {
                        "event": "all_audits_complete",
                        "summary_sha256": sha(
                            (args.output / "summary.json").read_bytes()
                        ),
                        "cases": 20,
                    }
                ),
                flush=True,
            )
            return
        if failures:
            return
        time.sleep(45)
    print(
        json.dumps({"event": "bounded_monitor_timeout", "archives": len(completed)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
