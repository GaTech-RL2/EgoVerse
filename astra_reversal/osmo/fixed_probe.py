"""Develop a faster reversible solver, then run explicitly agent-free pilots."""

import json
import subprocess
import sys
import time

from astra_reversal.osmo.experiment import RESULTS, ROOT, Archive, write_json
from astra_reversal.records import file_sha256


def main():
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.policies.pi05 import modeling_pi05  # noqa: F401

    assert torch.cuda.device_count() == 1
    assert "L40S" in torch.cuda.get_device_name(0)
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = Archive()

    def command(label, *args):
        began = time.perf_counter()
        print(json.dumps({"phase": label, "status": "started"}), flush=True)
        with (RESULTS / f"{label}.log").open("w") as stream:
            result = subprocess.run(
                [sys.executable, "-m", "astra_reversal", *args],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        archive.sync()
        print(
            json.dumps(
                {
                    "phase": label,
                    "returncode": result.returncode,
                    "seconds": time.perf_counter() - began,
                }
            ),
            flush=True,
        )
        if result.returncode:
            print((RESULTS / f"{label}.log").read_text()[-5000:], flush=True)
            raise RuntimeError(f"Phase {label} failed")

    try:
        inventory = json.loads(
            (
                ROOT / "astra_reversal/checkpoints/lerobot_pi05_libero_base.json"
            ).read_text()
        )
        checkpoint = ROOT / inventory["local_directory"]
        snapshot_download(
            inventory["repo_id"],
            revision=inventory["revision"],
            local_dir=checkpoint,
            token=False,
        )
        for entry in inventory["files"]:
            assert file_sha256(checkpoint / entry["path"]) == entry["sha256"]
        base = json.loads(
            (
                ROOT / "astra_reversal/configs/pi05_libero_openpi_inputs_baseline.json"
            ).read_text()
        )
        base["policy"]["device"] = "cuda"
        base["flow"]["time_power"] = 3.0
        base["evaluation"]["save_flow_traces"] = False
        write_json(RESULTS / "base.json", base)
        common = ["--libero-root", "astra_reversal/.deps/libero"]
        manifest = str(RESULTS / "development_manifest.json")
        command(
            "manifest",
            "make-manifest",
            "--config",
            str(RESULTS / "base.json"),
            *common,
            "--tasks",
            "0",
            "--trials",
            "1",
            "--output",
            manifest,
        )
        candidates = []
        for solver, resolutions in (
            ("heun", [50, 100, 200]),
            ("rk4", [20, 50, 100, 200]),
        ):
            config = json.loads(json.dumps(base))
            config["flow"]["integrator"] = solver
            config_path = RESULTS / f"{solver}.json"
            report_path = RESULTS / f"{solver}_diagnostics.json"
            write_json(config_path, config)
            command(
                f"{solver}_diagnostics",
                "diagnostics",
                "--config",
                str(config_path),
                "--manifest",
                manifest,
                *common,
                "--resolutions",
                *map(str, resolutions),
                "--action-atol",
                "0.02",
                "--noise-atol",
                "0.1",
                "--parity-atol",
                "0.00001",
                "--output",
                str(report_path),
            )
            report = json.loads(report_path.read_text())
            for row in report["results"]:
                print(
                    json.dumps(
                        {
                            "phase": "power_grid_result",
                            "solver": solver,
                            "steps": row["steps"],
                            "passed": row["passed"],
                            "noise": row["known_noise_recovery"],
                            "action": row["controller_reconstruction"],
                        }
                    ),
                    flush=True,
                )
                if row["passed"]:
                    candidates.append(
                        {
                            "solver": solver,
                            "steps": row["steps"],
                            "velocity_evaluations_per_solve": row["steps"]
                            * (2 if solver == "heun" else 4),
                            "diagnostics": str(report_path),
                        }
                    )
        if not candidates:
            write_json(
                RESULTS / "status.json",
                {
                    "status": "no_passing_fixed_solver",
                    "control_pilots_run": False,
                    "astra_calls_run": False,
                },
            )
            return
        # Predeclared development selection: least evaluations per solve, then N.
        chosen = min(
            candidates,
            key=lambda row: (row["velocity_evaluations_per_solve"], row["steps"]),
        )
        write_json(
            RESULTS / "selected_solver.json",
            {
                **chosen,
                "time_power": 3.0,
                "selection_rule": "Minimum function evaluations among passing development diagnostics; no control-success selection",
            },
        )
        comparisons = {}
        for method in (
            "policy_fresh",
            "policy_reused",
            "inversion_only",
            "compute_matched",
        ):
            config = json.loads(json.dumps(base))
            config["method"] = method
            config["flow"].update(
                integrator=chosen["solver"],
                inversion_steps=chosen["steps"],
                generation_steps=chosen["steps"],
            )
            config["controller"]["compute_matched_candidates"] = 3
            config_path = RESULTS / f"{method}.json"
            write_json(config_path, config)
            diagnostic_args = (
                ["--diagnostics", chosen["diagnostics"]]
                if method == "inversion_only"
                else []
            )
            command(
                method,
                "run",
                "--config",
                str(config_path),
                "--manifest",
                manifest,
                *common,
                *diagnostic_args,
                "--output",
                str(RESULTS / method),
            )
            comparisons[method] = json.loads(
                (RESULTS / method / "summary.json").read_text()
            )
            write_json(
                RESULTS / "pilot_comparison.json",
                {
                    "scope": "One paired development episode per method; no Astra or benchmark improvement claim",
                    "selected_solver": chosen,
                    "methods": comparisons,
                },
            )
        write_json(
            RESULTS / "status.json",
            {
                "status": "complete",
                "control_pilots_run": True,
                "astra_calls_run": False,
            },
        )
    except Exception as exc:
        write_json(
            RESULTS / "status.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
