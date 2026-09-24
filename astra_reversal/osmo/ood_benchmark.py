"""Evaluate both published LIBERO-OOD suites on frozen, paired reset states."""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict

from astra_reversal.config import BenchmarkConfig, RunConfig
from astra_reversal.evaluate import read_episodes, summarize
from astra_reversal.osmo.experiment import RESULTS, ROOT, Archive, write_json
from astra_reversal.records import file_sha256

SUITES = ("libero_goal_ood", "libero_spatial_ood")
LIBERO = "astra_reversal/.deps/libero-ood/third_party/modified_libero"


def main():
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.policies.pi05 import modeling_pi05  # noqa: F401

    count = int(os.environ.get("ASTRA_BENCHMARK_GPUS", "8"))
    assert count >= 2 and count % 2 == 0
    assert torch.cuda.device_count() == count
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    assert all("L40S" in name for name in names)
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = Archive()
    processes, streams, jobs = [], [], []
    runtime = {
        "workflow": os.environ["ASTRA_RUN_ID"],
        "gpu_names": names,
        "payload_sha256": os.environ["PAYLOAD_SHA256"],
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "numpy",
                "mujoco",
                "robosuite",
                "transformers",
                "lerobot",
            )
        },
        "paper": "https://arxiv.org/abs/2505.03500v5",
        "protocol_notes": [
            "Released modified LIBERO revision, success predicates, ten tasks per suite, ten trials, 300 action steps plus 10 stabilization steps.",
            "Released default environment seed 7; each task's reset stream captured before evaluation and shared across all paired methods.",
            "Frozen LeRobot pi05 weights with verified OpenPI LIBERO inputs; this is a new pi05 baseline, not a replication of the paper's pi0 checkpoint.",
            "Five actions executed per ten-action chunk, as specified in Astra SPEC; policy noise uses SeedSequence([seed, step, stream]) and repeats its schedule across episodes.",
            "No Astra calls in this baseline; no test-success-based solver or prompt tuning.",
        ],
    }
    write_json(RESULTS / "runtime.json", runtime)
    print(json.dumps({"phase": "allocation_verified", **runtime}), flush=True)
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

        per_suite = count // 2
        manifests = {}
        for suite_index, suite in enumerate(SUITES):
            directory = RESULTS / suite
            config = json.loads(
                (
                    ROOT
                    / "astra_reversal/configs/pi05_libero_openpi_inputs_baseline.json"
                ).read_text()
            )
            config["seed"] = 7
            config["benchmark"] = asdict(BenchmarkConfig.preset(suite))
            config["evaluation"].update(split="test", save_flow_traces=False)
            for shard in range(per_suite):
                config["policy"]["device"] = f"cuda:{suite_index * per_suite + shard}"
                RunConfig.from_dict(config).validate()
                write_json(directory / f"config_{shard}.json", config)
            manifest = directory / "test_manifest.json"
            with (directory / "manifest.log").open("w") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "astra_reversal",
                        "make-manifest",
                        "--config",
                        str(directory / "config_0.json"),
                        "--libero-root",
                        LIBERO,
                        "--output",
                        str(manifest),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            frozen = json.loads(manifest.read_text())
            assert len(frozen["episodes"]) == 100
            assert not any(e["initially_successful"] for e in frozen["episodes"])
            manifests[suite] = frozen
            print(
                json.dumps(
                    {
                        "phase": "manifest_frozen",
                        "suite": suite,
                        "episodes": 100,
                        "sha256": frozen["sha256"],
                    }
                ),
                flush=True,
            )
            for shard in range(per_suite):
                jobs.append((suite, shard, directory, manifest))
        archive.sync()

        for suite, shard, directory, manifest in jobs:
            stream = (directory / f"shard_{shard}.log").open("w")
            streams.append(stream)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "astra_reversal",
                        "run",
                        "--config",
                        str(directory / f"config_{shard}.json"),
                        "--manifest",
                        str(manifest),
                        "--libero-root",
                        LIBERO,
                        "--shard-index",
                        str(shard),
                        "--num-shards",
                        str(per_suite),
                        "--output",
                        str(directory / f"shard_{shard}"),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            )
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(
                    "An OOD evaluation shard failed; inspect its archived log"
                )
            progress = {"phase": "evaluation", "total": 200, "suites": {}}
            for suite in SUITES:
                completed = []
                for candidate, shard, directory, _ in jobs:
                    path = directory / f"shard_{shard}" / "events.jsonl"
                    if candidate != suite or not path.exists():
                        continue
                    for line in path.read_text().splitlines():
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event["kind"] == "episode_end":
                            completed.append(event)
                progress["suites"][suite] = {
                    "completed": len(completed),
                    "total": 100,
                    "successes": sum(bool(x["success"]) for x in completed),
                    "execution_errors": sum(
                        x.get("failure") is not None for x in completed
                    ),
                }
            progress["completed"] = sum(
                s["completed"] for s in progress["suites"].values()
            )
            write_json(RESULTS / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            archive.sync()
            time.sleep(30)
        assert all(p.wait() == 0 for p in processes)
        report = {
            "workflow": os.environ["ASTRA_RUN_ID"],
            "method": "policy_fresh",
            "suites": {},
        }
        all_episodes = []
        for suite in SUITES:
            episodes = [
                e
                for shard in range(per_suite)
                for e in read_episodes(RESULTS / suite / f"shard_{shard}")
            ]
            ids = [e["episode_id"] for e in episodes]
            assert len(ids) == len(set(ids)) == 100
            assert set(ids) == {e["episode_id"] for e in manifests[suite]["episodes"]}
            summary = summarize(episodes)
            summary.update(
                successes=sum(bool(e["success"]) for e in episodes),
                execution_errors=sum(e.get("failure") is not None for e in episodes),
                task_manifest_sha256=manifests[suite]["sha256"],
            )
            report["suites"][suite] = summary
            write_json(RESULTS / suite / "summary.json", summary)
            write_json(RESULTS / suite / "episodes.json", episodes)
            all_episodes.extend(episodes)
        report.update(
            episodes=200,
            successes=sum(bool(e["success"]) for e in all_episodes),
            execution_errors=sum(e.get("failure") is not None for e in all_episodes),
            astra_calls_run=False,
            interpretation="New frozen pi05 baseline on the paper's 20 released OOD tasks",
        )
        report["success_rate"] = report["successes"] / 200
        write_json(RESULTS / "summary.json", report)
        write_json(RESULTS / "episodes.json", all_episodes)
        write_json(
            RESULTS / "status.json",
            {
                "status": "complete"
                if not report["execution_errors"]
                else "completed_with_execution_errors",
                **report,
            },
        )
        print(
            json.dumps(
                {
                    "phase": "complete",
                    "episodes": 200,
                    "successes": report["successes"],
                    "execution_errors": report["execution_errors"],
                }
            ),
            flush=True,
        )
    except Exception as exc:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        write_json(
            RESULTS / "status.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        for stream in streams:
            stream.close()
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
