"""Measure native flow invertibility with adaptive reference numerics on L40S."""

import json
import os
import subprocess
import sys

import numpy as np

from astra_reversal.__main__ import enable_local_openpi, load_config
from astra_reversal.action_adapter import ActionAdapter, ActionSpec
from astra_reversal.adaptive_probe import integrate_reference
from astra_reversal.flow import error_metrics
from astra_reversal.libero_runner import LiberoEpisode, configure_libero
from astra_reversal.osmo.experiment import RESULTS, ROOT, Archive, write_json
from astra_reversal.policy_adapter import load_policy
from astra_reversal.records import digest, file_sha256, to_numpy


def main():
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.policies.pi05 import modeling_pi05  # noqa: F401

    assert torch.cuda.device_count() == 1
    assert "L40S" in torch.cuda.get_device_name(0)
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive, episode = Archive(), None
    report = {
        "workflow": os.environ["ASTRA_RUN_ID"],
        "payload_sha256": os.environ["PAYLOAD_SHA256"],
        "purpose": "Development-only adaptive reference integration; no Astra or success claim",
        "velocity_precision": "float32",
        "accumulation_precision": "float64",
        "tolerances": {"action_atol": 0.02, "noise_atol": 0.1, "parity_atol": 1e-5},
        "results": [],
    }
    try:
        enable_local_openpi()
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
        config_dict = json.loads(
            (
                ROOT / "astra_reversal/configs/pi05_libero_openpi_inputs_baseline.json"
            ).read_text()
        )
        config_dict["policy"]["device"] = "cuda"
        config_path = RESULTS / "config.json"
        write_json(config_path, config_dict)
        config = load_config(config_path)
        manifest_path = RESULTS / "development_manifest.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "astra_reversal",
                "make-manifest",
                "--config",
                str(config_path),
                "--libero-root",
                "astra_reversal/.deps/libero",
                "--tasks",
                "0",
                "--trials",
                "1",
                "--output",
                str(manifest_path),
            ],
            check=True,
        )
        manifest = json.loads(manifest_path.read_text())
        _, create = configure_libero(
            "astra_reversal/.deps/libero", config.benchmark, RESULTS / "libero_config"
        )
        policy = load_policy(
            config.policy.checkpoint,
            config.policy.config_name,
            config.policy.device,
            config.policy.checkpoint_provenance,
            config.policy.training_overlap,
            tokenizer_path=config.policy.tokenizer_path,
            input_profile=config.policy.input_profile,
            reference_assets=config.policy.reference_assets,
        )
        entry = manifest["episodes"][0]
        env, _, _ = create(entry["task_id"], entry["seed"])
        episode = LiberoEpisode(
            env,
            entry,
            config.benchmark,
            policy_image_size=policy.observation_image_size,
        )
        spec = ActionSpec.from_environment(env, policy.horizon, policy.action_dim)
        actions = ActionAdapter(spec, policy.input_transform, policy.output_transform)
        observation = episode.observe()
        condition = policy.prepare(
            observation, digest(observation), entry["instruction"]
        )
        noise = policy.noise(np.random.default_rng(config.seed))
        source = to_numpy(noise)
        report.update(
            checkpoint=policy.metadata,
            task_manifest_sha256=manifest["sha256"],
            action_spec_id=spec.action_spec_id,
            condition_id=condition.condition_id,
            seed=config.seed,
        )
        baseline = policy.sample(condition, noise, steps=10)
        parity = error_metrics(
            policy.reference_actions(condition, noise, steps=10),
            actions.output_transform({"actions": to_numpy(baseline.value)[0]})[
                "actions"
            ],
        )
        report["native_euler_parity"] = parity
        assert parity["max_abs"] <= report["tolerances"]["parity_atol"]
        np.savez_compressed(RESULTS / "input.npz", noise=source, **observation)

        def velocity(x, t):
            return to_numpy(condition.velocity(policy.tensor(x), t))

        def channel_metrics(reference, actual):
            return {
                "full": error_metrics(reference, actual),
                "action_channels": error_metrics(reference[..., :7], actual[..., :7]),
                "padding_channels": error_metrics(reference[..., 7:], actual[..., 7:]),
            }

        for index, (method, rtol) in enumerate(
            [
                ("RK45", 1e-3),
                ("RK45", 1e-4),
                ("RK45", 1e-5),
                ("RK45", 1e-6),
                ("DOP853", 1e-6),
            ]
        ):
            kwargs = dict(rtol=rtol, atol=rtol / 100, method=method)
            print(json.dumps({"phase": "adaptive_solve", **kwargs}), flush=True)
            forward = integrate_reference(velocity, source, start=1, end=0, **kwargs)
            # Runtime endpoints are float32; test that practical precision here.
            endpoint = forward.value.astype(np.float32)
            inverse = integrate_reference(velocity, endpoint, start=0, end=1, **kwargs)
            recovered = inverse.value.astype(np.float32)
            replay = integrate_reference(velocity, recovered, start=1, end=0, **kwargs)
            reconstruction = channel_metrics(endpoint, replay.value)
            recovery = channel_metrics(source, recovered)
            decoded = [
                actions.output_transform({"actions": x[0]})["actions"]
                for x in (endpoint, replay.value)
            ]
            controller_error = error_metrics(*decoded)
            row = {
                **kwargs,
                "endpoint_precision": "float32",
                "reconstruction": reconstruction,
                "known_noise_recovery": recovery,
                "controller_reconstruction": controller_error,
                "solves": {
                    name: {
                        "grid": solve.grid,
                        "velocity_evaluations": solve.velocity_evaluations,
                        "latency_seconds": solve.latency_seconds,
                    }
                    for name, solve in (
                        ("forward", forward),
                        ("inverse", inverse),
                        ("replay", replay),
                    )
                },
                "passed": reconstruction["full"]["max_abs"] <= 0.02
                and controller_error["max_abs"] <= 0.02
                and recovery["full"]["max_abs"] <= 0.1,
            }
            np.savez_compressed(
                RESULTS / f"solve_{index}.npz",
                noise=source,
                endpoint=endpoint,
                recovered=recovered,
                replay=replay.value,
            )
            report["results"].append(row)
            write_json(RESULTS / "reference_numerics.json", report)
            archive.sync()
            print(
                json.dumps(
                    {
                        "phase": "adaptive_result",
                        "method": method,
                        "rtol": rtol,
                        "noise": recovery,
                        "action": controller_error,
                        "passed": row["passed"],
                    }
                ),
                flush=True,
            )
        write_json(
            RESULTS / "status.json", {"status": "complete", "astra_calls_run": False}
        )
    except Exception as exc:
        write_json(
            RESULTS / "status.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        if episode is not None:
            episode.close()
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
