"""Unscored native reset/replay validation, with no model calls."""

import json
import os
from pathlib import Path

from astra_reversal.hardware_interface.common import file_hash, write_json
from astra_reversal.hardware_interface.libero import Environment, array_hash, configure
from astra_reversal.hardware_interface.osmo_worker import archive


def main():
    root = Path("artifacts/reset-validation")
    root.mkdir(parents=True, exist_ok=False)
    reference_path = Path(os.environ["HARDWARE_RESET_REFERENCE"])
    reference = json.loads(reference_path.read_text())
    prior = {(r["task_id"], r["init_state_index"]): r for r in reference["cases"]}
    registry = configure("upstream-libero", root / "libero-config")
    report = {
        "scope": "Unscored reset and deterministic native-action replay; zero model calls",
        "reference_sha256": file_hash(reference_path),
        "source_commit": os.environ["HARDWARE_SOURCE_COMMIT"],
        "cases": [],
    }
    actions = [[0, 0, 0.1, 0, 0, 0, -1]] * 2 + [[0.1, 0, 0, 0, 0, 0, -1]] * 3
    actions += [[-0.1, 0, 0, 0, 0, 0, -1]] * 3
    try:
        for task, index in ((5, 1), (1, 0), (5, 3), (5, 2)):
            repetitions = []
            for repeat in range(2):
                env = Environment(
                    registry, "upstream-libero", "libero_10", task, index, 137
                )
                try:
                    previous = prior[task, index]
                    assert (
                        env.reset_receipt["setter_state_sha256"]
                        == previous["setter"][0]["actual_sha256"]
                    )
                    if previous["status"] == "passed":
                        assert all(
                            env.reset_receipt[k] == v
                            for k, v in previous["reset"].items()
                        )
                    states = [array_hash(env.env.get_sim_state())]
                    for action in actions:
                        env.step(action)
                        states.append(array_hash(env.env.get_sim_state()))
                    repetitions.append(
                        {"reset": env.reset_receipt, "state_trace_sha256": states}
                    )
                    env.terminal_snapshot(root / (f"t{task}-i{index}-r{repeat}.npz"))
                finally:
                    env.close()
            assert repetitions[0] == repetitions[1]
            report["cases"].append(
                {
                    "task_id": task,
                    "init_state_index": index,
                    "status": "passed",
                    "repetitions": repetitions,
                }
            )
            write_json(root / "report.json", report, replace=True)
            print(
                json.dumps(
                    {
                        "reset_validation_case": [task, index],
                        "status": "passed",
                        "match": repetitions[0]["reset"]["setter_validation"],
                    }
                ),
                flush=True,
            )
        report["status"] = "passed"
        write_json(root / "report.json", report, replace=True)
        print("_NATIVE_RESET_VALIDATION_=" + json.dumps(report), flush=True)
    finally:
        archive(root, os.environ["HARDWARE_WORKFLOW"])


if __name__ == "__main__":
    main()
