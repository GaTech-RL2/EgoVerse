"""Run the complete frozen 500-episode baseline across an allocated GPU node."""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

from astra_reversal.evaluate import read_episodes, summarize
from astra_reversal.osmo.experiment import RESULTS, ROOT, Archive, write_json
from astra_reversal.records import file_sha256


def main():
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.policies.pi05 import modeling_pi05  # noqa: F401

    count = int(os.environ.get("ASTRA_BENCHMARK_GPUS", "8"))
    assert torch.cuda.device_count() == count
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    assert all("L40S" in name for name in names)
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = Archive()
    processes, streams = [], []
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

        config = json.loads(
            (
                ROOT / "astra_reversal/configs/pi05_libero_openpi_inputs_baseline.json"
            ).read_text()
        )
        config["evaluation"].update(split="test", save_flow_traces=False)
        for i in range(count):
            config["policy"]["device"] = f"cuda:{i}"
            write_json(RESULTS / f"config_{i}.json", config)
        manifest = RESULTS / "test_manifest.json"
        common = ["--libero-root", "astra_reversal/.deps/libero"]
        with (RESULTS / "manifest.log").open("w") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "astra_reversal",
                    "make-manifest",
                    "--config",
                    str(RESULTS / "config_0.json"),
                    *common,
                    "--output",
                    str(manifest),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        frozen = json.loads(manifest.read_text())
        assert len(frozen["episodes"]) == 500
        print(
            json.dumps(
                {
                    "phase": "manifest_frozen",
                    "episodes": 500,
                    "sha256": frozen["sha256"],
                }
            ),
            flush=True,
        )
        archive.sync()

        for i in range(count):
            stream = (RESULTS / f"shard_{i}.log").open("w")
            streams.append(stream)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "astra_reversal",
                        "run",
                        "--config",
                        str(RESULTS / f"config_{i}.json"),
                        "--manifest",
                        str(manifest),
                        *common,
                        "--shard-index",
                        str(i),
                        "--num-shards",
                        str(count),
                        "--output",
                        str(RESULTS / f"shard_{i}"),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            )

        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError(
                    "An evaluation shard exited with an error; inspect its log"
                )
            completed = []
            for i in range(count):
                path = RESULTS / f"shard_{i}" / "events.jsonl"
                if path.exists():
                    for line in path.read_text().splitlines():
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # A currently appended progress snapshot.
                        if event["kind"] == "episode_end":
                            completed.append(event)
            progress = {
                "phase": "evaluation",
                "completed": len(completed),
                "total": 500,
                "successes": sum(bool(x["success"]) for x in completed),
                "execution_errors": sum(
                    x.get("failure") is not None for x in completed
                ),
            }
            write_json(RESULTS / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            archive.sync()
            time.sleep(30)
        assert all(process.wait() == 0 for process in processes)
        for stream in streams:
            stream.close()
        episodes = [
            episode
            for i in range(count)
            for episode in read_episodes(RESULTS / f"shard_{i}")
        ]
        expected_ids = {entry["episode_id"] for entry in frozen["episodes"]}
        actual_ids = [episode["episode_id"] for episode in episodes]
        assert len(actual_ids) == len(set(actual_ids)) == 500
        assert set(actual_ids) == expected_ids
        report = summarize(episodes)
        report.update(
            successes=sum(bool(x["success"]) for x in episodes),
            execution_errors=sum(x.get("failure") is not None for x in episodes),
            task_manifest_sha256=frozen["sha256"],
            input_profile="openpi_libero",
            workflow=os.environ["ASTRA_RUN_ID"],
            interpretation="Full standard LIBERO-10 evaluation of the selected LeRobot weights with OpenPI input settings; no Astra calls",
        )
        write_json(RESULTS / "summary.json", report)
        write_json(RESULTS / "episodes.json", episodes)
        write_json(
            RESULTS / "status.json",
            {
                "status": "complete"
                if not report["execution_errors"]
                else "completed_with_execution_errors",
                "episodes": 500,
                "successes": report["successes"],
                "execution_errors": report["execution_errors"],
                "astra_calls_run": False,
            },
        )
        print(
            json.dumps(
                {
                    "phase": "complete",
                    "successes": report["successes"],
                    "episodes": 500,
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
