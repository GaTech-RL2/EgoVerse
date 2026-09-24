"""Replace the complete baseline shard affected by a CUDA launch failure."""

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

from astra_reversal.config import BenchmarkConfig, RunConfig
from astra_reversal.evaluate import read_episodes, summarize
from astra_reversal.libero_runner import load_task_manifest, shard_entries
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
    process = None
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
        for item in inventory["files"]:
            assert file_sha256(checkpoint / item["path"]) == item["sha256"]
        config = json.loads(
            (
                ROOT / "astra_reversal/configs/pi05_libero_openpi_inputs_baseline.json"
            ).read_text()
        )
        config["seed"] = 7
        config["benchmark"] = asdict(BenchmarkConfig.preset("libero_goal_ood"))
        config["policy"]["device"] = "cuda"
        config["evaluation"].update(split="test", save_flow_traces=False)
        config_path = RESULTS / "config.json"
        write_json(config_path, config)
        source = ROOT / "astra_reversal/.deps/ood-inputs/libero_goal_ood_manifest.json"
        manifest = load_task_manifest(source, RunConfig.from_dict(config))
        assert (
            manifest["sha256"]
            == "f9b51a9929093f8f50f6e4106ff0da562f66c287b8583c569d719fc7b8ffad5f"
        )
        manifest_path = RESULTS / "test_manifest.json"
        manifest_path.write_bytes(source.read_bytes())
        expected = {entry["episode_id"] for entry in shard_entries(manifest, 3, 4)}
        runtime = {
            "workflow": os.environ["ASTRA_RUN_ID"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "replaces_workflow": "astra-pi05-ood-baseline-20260924-1",
            "replaces_suite": "libero_goal_ood",
            "replaces_shard": 3,
            "num_shards": 4,
            "replacement_scope": "All 25 episodes in the affected shard, including previously successful episodes; no outcome-based replacement",
            "reason": "Original GPU shard encountered CUDA unspecified launch failure at task7 state1, then seven subsequent episodes failed on the poisoned CUDA context",
            "task_manifest_sha256": manifest["sha256"],
        }
        write_json(RESULTS / "runtime.json", runtime)
        with (RESULTS / "repair.log").open("w") as stream:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "astra_reversal",
                    "run",
                    "--config",
                    str(config_path),
                    "--manifest",
                    str(manifest_path),
                    "--libero-root",
                    "astra_reversal/.deps/libero-ood/third_party/modified_libero",
                    "--shard-index",
                    "3",
                    "--num-shards",
                    "4",
                    "--output",
                    str(RESULTS / "repair"),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                archive.sync()
                time.sleep(30)
            if process.wait() != 0:
                raise RuntimeError("Replacement baseline shard exited with an error")
        episodes = read_episodes(RESULTS / "repair")
        assert len(episodes) == len({e["episode_id"] for e in episodes}) == 25
        assert {e["episode_id"] for e in episodes} == expected
        report = {
            **summarize(episodes),
            **runtime,
            "successes": sum(bool(e["success"]) for e in episodes),
            "execution_errors": sum(e.get("failure") is not None for e in episodes),
        }
        write_json(RESULTS / "summary.json", report)
        write_json(RESULTS / "episodes.json", episodes)
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
                    "episodes": 25,
                    "successes": report["successes"],
                    "execution_errors": report["execution_errors"],
                }
            ),
            flush=True,
        )
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        write_json(
            RESULTS / "status.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
