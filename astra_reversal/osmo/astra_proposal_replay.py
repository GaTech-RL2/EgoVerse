"""Characterize fixed RK4 resolutions on recorded genuine development proposals.

This probe makes no Astra calls, runs no environment episodes, and selects no
solver. ``--prepare-inputs`` streams only the planned NPY members from an archive;
``--verify-inputs-only`` checks the resulting bundle without loading a policy.
"""

import argparse
import copy
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from astra_reversal.action_adapter import ActionAdapter, ActionSpec
from astra_reversal.agent import CAMERAS, parse_proposal
from astra_reversal.config import RunConfig
from astra_reversal.flow import error_metrics
from astra_reversal.osmo.experiment import RESULTS, ROOT, Archive
from astra_reversal.osmo.runtime_probe import RecordedArrays, write_json
from astra_reversal.records import digest, file_sha256, to_numpy

MODEL = "azure/openai/gpt-6-astra"
STEPS = (100, 200, 500)
CASE_STEPS = (310, 330, 485)
OPTIONS = {"time_power": 3.0}
ACTION_ATOL = 0.02
SOURCE_FILES = {
    "manifest.json": "reversal/manifest.json",
    "events.jsonl": "reversal/events.jsonl",
    "provider.jsonl": "astra_provider.jsonl",
    "audit.json": "runtime_inversion_audit.json",
    "live_proposal_provenance.json": "live_proposal_provenance.json",
    "worker_summary.json": "development_smoke.json",
    "worker_runtime.json": "runtime.json",
    "worker_config.json": "frozen/development_config.json",
    "task_manifest.json": "frozen/development_manifest.json",
    "frozen_plan.json": "frozen_plan.json",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def prepare_inputs(source, plan_path, catalog_path, destination):
    """Retain only allowlisted regular files; never copy or print signed URLs."""
    source, destination = Path(source), Path(destination)
    plan = read_json(plan_path)
    require(plan["source_split"] == "development", "Development inputs required")
    for item in plan["source_records"]:
        require(
            file_sha256(source / item["path"]) == item["sha256"],
            f"Source record changed: {item['path']}",
        )
    expected = {}
    for case in plan["cases"]:
        for reference in case["arrays"].values():
            name = reference["archive_member"]
            require(
                name == "results/reversal/" + reference["array"]
                and Path(reference["array"]).parts[0] == "arrays"
                and ".." not in Path(name).parts,
                "Unsafe planned array member",
            )
            if name in expected:
                require(expected[name] == reference, "Conflicting member references")
            expected[name] = reference
    require(len(expected) == 21, "Expected exactly 21 planned array members")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name, relative in SOURCE_FILES.items():
            shutil.copyfile(source / relative, destination / name)
        shutil.copyfile(plan_path, destination / "replay_plan.json")
        catalog = read_json(catalog_path)
        hasher, transferred, found = hashlib.sha256(), 0, set()

        class HashingReader:
            def read(self, size=-1):
                nonlocal transferred
                data = response.read(size)
                hasher.update(data)
                transferred += len(data)
                return data

        try:
            with urllib.request.urlopen(
                catalog["artifacts.tar.gz"], timeout=60
            ) as response:
                etag = response.headers.get("ETag")
                require(
                    etag == plan["input_extraction"]["source_archive_etag"],
                    "Completed archive ETag changed",
                )
                reader = HashingReader()
                with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                    for member in archive:
                        if member.name not in expected:
                            continue
                        require(
                            member.isfile() and member.name not in found,
                            "Planned archive member is duplicated or not a regular file",
                        )
                        reference = expected[member.name]
                        value_bytes = archive.extractfile(member).read()
                        value = np.load(io.BytesIO(value_bytes), allow_pickle=False)
                        require(
                            digest(value) == reference["sha256"]
                            and list(value.shape) == reference["shape"]
                            and str(value.dtype) == reference["dtype"],
                            "Extracted array differs from its recorded identity",
                        )
                        path = destination / reference["array"]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(value_bytes)
                        found.add(member.name)
                while reader.read(1024 * 1024):
                    pass
        except Exception as exc:
            # urllib exceptions may include a private presigned URL.
            raise RuntimeError(
                f"Archive transfer or verification failed ({type(exc).__name__})"
            ) from None
        require(found == set(expected), "Archive is missing planned arrays")
        require(
            transferred == plan["input_extraction"]["source_archive_bytes"],
            "Completed archive size changed",
        )
        inventory = {
            str(path.relative_to(destination)): {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        }
        bundle = {
            "schema_version": "1.0",
            "purpose": "Genuine Stage 1 development endpoint replay; no solver selection",
            "source_workflow": plan["source_workflow"],
            "source_payload_sha256": plan["source_payload_sha256"],
            "source_archive": {
                "sha256": hasher.hexdigest(),
                "bytes": transferred,
                "etag": etag,
            },
            "array_members": sorted(found),
            "files": inventory,
        }
        bundle["bundle_sha256"] = digest(bundle)
        write_json(destination / "bundle.json", bundle)
        loaded = load_inputs(destination)
        return loaded["provenance"]
    except BaseException:
        shutil.rmtree(destination)
        raise


def load_inputs(directory):
    """Bind exact arrays to original development events and actual provider output."""
    directory = Path(directory).resolve()
    bundle = read_json(directory / "bundle.json")
    require(
        bundle["schema_version"] == "1.0"
        and digest({k: v for k, v in bundle.items() if k != "bundle_sha256"})
        == bundle["bundle_sha256"],
        "Input bundle manifest digest mismatch",
    )
    actual_files = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }
    require(actual_files == {*bundle["files"], "bundle.json"}, "Unexpected input files")
    for name, item in bundle["files"].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory), "Input path escapes bundle")
        require(
            file_sha256(path) == item["sha256"]
            and path.stat().st_size == item["bytes"],
            f"Input file integrity mismatch: {name}",
        )
    manifest = read_json(directory / "manifest.json")
    config = RunConfig.from_dict(manifest["config"])
    plan = read_json(directory / "replay_plan.json")
    audit = read_json(directory / "audit.json")
    runtime = read_json(directory / "worker_runtime.json")
    summary = read_json(directory / "worker_summary.json")
    frozen = read_json(directory / "frozen_plan.json")
    task_manifest = read_json(directory / "task_manifest.json")
    require(
        config.method == "reversal"
        and config.benchmark.suite == "libero_10"
        and config.evaluation.split == "development"
        and config.policy.input_profile == "openpi_libero"
        and config.flow.integrator == "rk4"
        and config.flow.inversion_steps == config.flow.generation_steps == 100
        and config.flow.noise_mix_rho == 0.0
        and config.flow.solver_options == OPTIONS
        and runtime["tf32"] is False
        and summary["episode_outcome"]["split"] == "development"
        and summary["episode_outcome"]["suite"] == "libero_10"
        and plan["source_split"] == "development"
        and [row["observation_step"] for row in plan["cases"]] == list(CASE_STEPS)
        and plan["gpu_probe_plan"]["candidate_steps"] == list(STEPS),
        "Inputs differ from the declared development-only replay",
    )
    require(
        manifest["config"] == read_json(directory / "worker_config.json")
        and runtime["workflow"] == bundle["source_workflow"] == plan["source_workflow"]
        and runtime["payload_sha256"]
        == bundle["source_payload_sha256"]
        == plan["source_payload_sha256"]
        and manifest["task_manifest_sha256"] == task_manifest["sha256"]
        and task_manifest["sha256"] == frozen["task_manifest_sha256"]
        and task_manifest["sha256"]
        == digest({k: v for k, v in task_manifest.items() if k != "sha256"})
        and frozen["plan_sha256"]
        == digest({k: v for k, v in frozen.items() if k != "plan_sha256"}),
        "Development worker configuration or manifest binding failed",
    )
    run = audit["runs"][0]
    require(
        audit["status"] == "complete"
        and audit["action_atol"] == ACTION_ATOL
        and len(audit["runs"]) == 1
        and run["provenance"]["events_sha256"]
        == file_sha256(directory / "events.jsonl")
        and run["provenance"]["manifest_sha256"]
        == file_sha256(directory / "manifest.json")
        and summary["runtime_inversion_audit_sha256"]
        == file_sha256(directory / "audit.json"),
        "Recorded audit is not bound to these source files",
    )
    for item in plan["source_records"]:
        name = next(k for k, v in SOURCE_FILES.items() if v == item["path"])
        require(
            file_sha256(directory / name) == item["sha256"],
            "Planned source hash differs",
        )
    spec = ActionSpec(**manifest["action_spec"])
    require(
        spec.action_spec_id
        == digest(
            {
                k: manifest["action_spec"][k]
                for k in (
                    "horizon",
                    "model_action_dim",
                    "timestep_seconds",
                    "lower",
                    "upper",
                    "semantics",
                )
            }
        )
        and (spec.horizon, spec.model_action_dim) == (10, 32),
        "Controller action specification mismatch",
    )
    events, providers = (
        read_jsonl(directory / "events.jsonl"),
        read_jsonl(directory / "provider.jsonl"),
    )
    require(
        [e["sequence"] for e in events] == list(range(len(events))),
        "Noncontiguous events",
    )
    arrays, cases = RecordedArrays(directory), []
    for selected in plan["cases"]:
        source = next(
            row for row in run["plans"] if row["plan_id"] == selected["plan_id"]
        )
        event = events[source["agent_response_sequence"]]
        inverse = events[source["inverse_flow_sequence"]]
        forward = events[source["same_condition_roundtrip"]["generation_sequence"]]
        values = {key: arrays.read(ref) for key, ref in selected["arrays"].items()}
        observation = {key: values[key] for key in (*CAMERAS, "observation/state")}
        require(
            digest(observation)
            == selected["observation_id"]
            == source["inverse_observation_id"]
            and digest({**observation, "prompt": selected["prompt"]})
            == selected["condition_id"]
            == inverse["condition_id"]
            == forward["condition_id"]
            and inverse["observation_step"]
            == forward["observation_step"]
            == selected["observation_step"]
            and inverse["input"]["sha256"] == digest(values["encoded_endpoint"])
            and inverse["output"]["sha256"]
            == forward["input"]["sha256"]
            == digest(values["recorded_recovered_latent"])
            and forward["output"]["sha256"]
            == digest(values["recorded_first_generation"])
            and event["accepted"] is True,
            "Selected inverse/first-generation condition or tensor binding failed",
        )
        request = copy.deepcopy(event["request"])
        request["observation"] = observation
        fingerprint = request["request_fingerprint"]
        require(
            digest({k: v for k, v in request.items() if k != "request_fingerprint"})
            == fingerprint
            == selected["request_fingerprint"]
            and request["episode_id"]
            == selected["episode_id"]
            == plan["source_episode_id"]
            == summary["episode_outcome"]["episode_id"]
            and request["observation_step"] == selected["observation_step"]
            and request["stage"] == 1
            and request["task_instruction"] == selected["prompt"]
            and request["action_spec"] == manifest["action_spec"],
            "Original request fingerprint or specification mismatch",
        )
        matches = [
            row for row in providers if row["request_fingerprint"] == fingerprint
        ]
        require(len(matches) == 1, "Proposal lacks one unique provider response")
        provider = matches[0]
        raw = event["proposal"]["raw_response"]
        choices = provider["response"]["choices"]
        require(
            provider["accepted"] is True
            and provider["http_status"] == 200
            and provider["stage"] == 1
            and provider["requested_model"]
            == provider["response"]["model"]
            == event["proposal"]["model_version"]
            == MODEL
            and len(choices) == 1
            and choices[0]["finish_reason"] == "stop"
            and choices[0]["message"]["content"].strip() == raw.strip(),
            "Accepted proposal is not bound to genuine configured-model output",
        )
        proposal = parse_proposal(
            raw,
            request,
            model_version=MODEL,
            request_time=provider["request_time"],
            response_time=provider["response_time"],
            max_timeout=request["max_timeout_env_steps"],
            completion_types=request["supported_completion_types"],
        )
        require(
            np.array_equal(proposal.action_chunk, values["controller_actions"]),
            "Proposal values differ",
        )
        for key in (
            "encoded_endpoint",
            "recorded_recovered_latent",
            "recorded_first_generation",
        ):
            require(
                values[key].shape == (1, 10, 32) and values[key].dtype == np.float32,
                "Full float32 tensors required",
            )
        measured = error_metrics(
            values["encoded_endpoint"], values["recorded_first_generation"]
        )
        require(
            measured
            == selected["recorded_full_internal_reconstruction"]
            == source["same_condition_roundtrip"]["full_internal_reconstruction"]
            and (measured["max_abs"] <= ACTION_ATOL) == selected["recorded_passed"],
            "Extracted endpoints do not reproduce the recorded audit metrics",
        )
        cases.append({"source": selected, "observation": observation, "values": values})
    require(len(arrays.inventory) == 21, "Unexpected selected array count")
    require(
        [c["source"]["recorded_passed"] for c in cases] == [True, False, False],
        "Recorded development negative was not reproduced",
    )
    return {
        "manifest": manifest,
        "config": config,
        "spec": spec,
        "cases": cases,
        "provenance": {
            "bundle_sha256": bundle["bundle_sha256"],
            "bundle_file_sha256": file_sha256(directory / "bundle.json"),
            "source_workflow": bundle["source_workflow"],
            "source_payload_sha256": bundle["source_payload_sha256"],
            "source_archive": bundle["source_archive"],
            "files": bundle["files"],
            "array_members": bundle["array_members"],
            "recorded_metrics_recomputed_from_extracted_arrays": True,
        },
    }


def load_frozen_policy(inputs):
    from huggingface_hub import snapshot_download

    from astra_reversal.policy_adapter import load_policy

    config = inputs["config"].policy
    for inventory_name, location in (
        ("lerobot_pi05_libero_base.json", config.checkpoint),
        ("paligemma_tokenizer.json", config.tokenizer_path),
    ):
        inventory = read_json(ROOT / "astra_reversal/checkpoints" / inventory_name)
        if inventory_name.startswith("lerobot"):
            snapshot_download(
                inventory["repo_id"],
                revision=inventory["revision"],
                local_dir=ROOT / location,
                token=False,
            )
        for item in inventory["files"]:
            require(
                file_sha256(ROOT / location / item["path"]) == item["sha256"],
                "Pinned input asset changed",
            )
    client = ROOT / "external/openpi/packages/openpi-client/src"
    if client.is_dir():
        sys.path.insert(0, str(client))
    policy = load_policy(
        config.checkpoint,
        config.config_name,
        "cuda",
        config.checkpoint_provenance,
        config.training_overlap,
        tokenizer_path=config.tokenizer_path,
        input_profile=config.input_profile,
        reference_assets=config.reference_assets,
    )
    require(
        not any(p.requires_grad for p in policy.policy.parameters()),
        "Policy must remain frozen",
    )
    metadata, recorded = policy.metadata, inputs["manifest"]["checkpoint"]
    ignored = {"artifact", "requested_artifact", "device", "tokenizer_path"}
    require(
        {k: v for k, v in metadata.items() if k not in ignored}
        == {k: v for k, v in recorded.items() if k not in ignored},
        "Policy/input/source identity differs from recorded development execution",
    )
    return policy, ActionAdapter(
        inputs["spec"], policy.input_transform, policy.output_transform
    )


def replay_case(policy, actions, case, steps, output):
    source, values = case["source"], case["values"]
    observation, prompt = case["observation"], source["prompt"]
    encoded = actions.encode(
        values["controller_actions"], {**observation, "prompt": prompt}
    )
    require(
        np.array_equal(encoded, values["encoded_endpoint"]),
        "Live encoding differs from recorded full endpoint",
    )
    condition = policy.prepare(observation, source["observation_id"], prompt)
    require(
        condition.condition_id == source["condition_id"],
        "Prepared condition identity changed",
    )
    inverse = policy.invert(
        condition,
        encoded,
        steps=steps,
        solver="rk4",
        save_trace=False,
        **OPTIONS,
    )
    forward = policy.sample(
        condition,
        inverse.value,
        steps=steps,
        solver="rk4",
        save_trace=False,
        **OPTIONS,
    )
    latent, replay = to_numpy(inverse.value).copy(), to_numpy(forward.value).copy()
    state = to_numpy(condition.state)[0].copy()
    decoded = np.asarray(
        actions.output_transform({"actions": replay[0].copy(), "state": state})[
            "actions"
        ]
    )
    metrics = error_metrics(encoded, replay)
    controller = error_metrics(values["controller_actions"], decoded)
    artifact = (
        output / "arrays" / f"step_{source['observation_step']:04d}_rk4_{steps}.npz"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "encoded_endpoint": encoded,
        "recovered_latent": latent,
        "replayed_endpoint": replay,
        "decoded_replay": decoded,
    }
    np.savez_compressed(artifact, **tensors)
    result = {
        "case_id": source["case_id"],
        "plan_id": source["plan_id"],
        "observation_step": source["observation_step"],
        "condition_id": source["condition_id"],
        "steps": steps,
        "full_internal_reconstruction": metrics,
        "action_channel_reconstruction": error_metrics(
            encoded[..., :7], replay[..., :7]
        ),
        "padding_channel_reconstruction": error_metrics(
            encoded[..., 7:], replay[..., 7:]
        ),
        "direct_controller_reconstruction": controller,
        "passed_full_internal_tolerance": metrics["max_abs"] <= ACTION_ATOL,
        "passed_direct_controller_tolerance": controller["max_abs"] <= ACTION_ATOL,
        "clipping_applied_to_metrics": False,
        "known_noise_recovery": None,
        "solves": {
            "inverse": {
                "grid": inverse.grid,
                "velocity_evaluations": inverse.velocity_evaluations,
                "latency_seconds": inverse.latency_seconds,
            },
            "forward": {
                "grid": forward.grid,
                "velocity_evaluations": forward.velocity_evaluations,
                "latency_seconds": forward.latency_seconds,
            },
        },
        "velocity_evaluations": inverse.velocity_evaluations
        + forward.velocity_evaluations,
        "latency_seconds": inverse.latency_seconds + forward.latency_seconds,
        "artifacts": {
            "path": str(artifact.relative_to(output)),
            "file_sha256": file_sha256(artifact),
            "arrays": {
                key: {
                    "sha256": digest(value),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for key, value in tensors.items()
            },
        },
    }
    require(result["velocity_evaluations"] == 8 * steps, "Unexpected solve cost")
    if steps == 100:
        result["recorded_n100_reproduction"] = {
            "latent_difference": error_metrics(
                values["recorded_recovered_latent"], latent
            ),
            "endpoint_difference": error_metrics(
                values["recorded_first_generation"], replay
            ),
            "latent_bitwise_equal": bool(
                np.array_equal(values["recorded_recovered_latent"], latent)
            ),
            "endpoint_bitwise_equal": bool(
                np.array_equal(values["recorded_first_generation"], replay)
            ),
            "recorded_full_internal_reconstruction": source[
                "recorded_full_internal_reconstruction"
            ],
            "recorded_pass_fail_reproduced": result["passed_full_internal_tolerance"]
            == source["recorded_passed"],
        }
    return result


def private_download_catalog(archive):
    """Make an operational catalog outside the result tree; never log its URLs."""
    destination = Path("/osmo/run/workspace/astra-proposal-replay-download.json")
    catalog = {
        name: archive.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": "rldb", "Key": f"{archive.prefix}/{name}"},
            ExpiresIn=86400,
        )
        for name in (
            "astra_proposal_replay.json",
            "status.json",
            "progress.json",
            "bootstrap.log",
            "artifacts.tar.gz",
        )
    }
    with os.fdopen(
        os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
    ) as stream:
        json.dump(catalog, stream)


def run_probe(directory):
    import torch

    require(torch.cuda.device_count() == 1, "Exactly one allocated GPU is required")
    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_name(0)
    require("L40S" in gpu, "Replay requires an allocated L40S")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = load_inputs(directory)
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = Archive()
    began, exit_code = time.perf_counter(), 1
    report = {
        "schema_version": "development-proposal-replay-1.0",
        "status": "initializing",
        "solver": "rk4",
        "steps": list(STEPS),
        "solver_options": OPTIONS,
        "action_atol": ACTION_ATOL,
        "selected_solver": None,
        "astra_calls_run": False,
        "ood_inputs": 0,
        "environment_rollouts": 0,
        "known_noise_recovery": None,
        "interpretation": "Development endpoint characterization only; no solver selection or claim of a valid general Astra reversal method. Astra endpoints have no known generating policy noise.",
        "provenance": inputs["provenance"],
        "runtime": {
            "gpu": gpu,
            "device": "cuda",
            "tf32": False,
            "flow_dtype": "float32",
            "native_velocity_dtype": "float32",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "workflow": os.environ["ASTRA_RUN_ID"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "probe_source_sha256": file_sha256(Path(__file__)),
        },
        "results": [],
    }
    path = RESULTS / "astra_proposal_replay.json"
    try:
        shutil.copytree(directory, RESULTS / "frozen_input")
        shutil.copyfile(Path(__file__), RESULTS / "frozen_probe.py")
        private_download_catalog(archive)
        write_json(path, report)
        archive.sync()
        policy, actions = load_frozen_policy(inputs)
        report["checkpoint"] = policy.metadata
        report["action_spec_id"] = inputs["spec"].action_spec_id
        for steps in STEPS:
            for case in inputs["cases"]:
                progress = {
                    "phase": "replay",
                    "steps": steps,
                    "observation_step": case["source"]["observation_step"],
                    "completed_pairs": len(report["results"]),
                }
                print(json.dumps(progress), flush=True)
                write_json(RESULTS / "progress.json", progress)
                report["status"] = "running"
                result = replay_case(policy, actions, case, steps, RESULTS)
                report["results"].append(result)
                write_json(path, report)
                archive.sync()
                print(
                    json.dumps(
                        {
                            **progress,
                            "full_internal": result["full_internal_reconstruction"],
                            "direct_controller": result[
                                "direct_controller_reconstruction"
                            ],
                        }
                    ),
                    flush=True,
                )
        reproduction = [
            r["recorded_n100_reproduction"]
            for r in report["results"]
            if r["steps"] == 100
        ]
        report["n100_recorded_pass_fail_reproduced"] = all(
            r["recorded_pass_fail_reproduced"] for r in reproduction
        )
        report["n100_bitwise_reproduced"] = all(
            r["latent_bitwise_equal"] and r["endpoint_bitwise_equal"]
            for r in reproduction
        )
        report["velocity_evaluations"] = sum(
            r["velocity_evaluations"] for r in report["results"]
        )
        require(
            len(report["results"]) == 9 and report["velocity_evaluations"] == 19200,
            "Incomplete fixed replay sweep",
        )
        report["status"] = (
            "complete_reproduced"
            if report["n100_bitwise_reproduced"]
            else "complete_with_reproduction_differences"
        )
        exit_code = 0 if report["n100_bitwise_reproduced"] else 2
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - began
        report["exit_code"] = exit_code
        write_json(path, report)
        write_json(
            RESULTS / "status.json",
            {
                "status": report["status"],
                "exit_code": exit_code,
                "completed_pairs": len(report["results"]),
                "artifact_prefix": archive.prefix,
            },
        )
        archive.sync(include_arrays=True)
        if exit_code:
            Path("/tmp/astra-recovery-exit.status").write_text(str(exit_code) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "pairs": len(report["results"]),
                "velocity_evaluations": report["velocity_evaluations"],
            }
        ),
        flush=True,
    )
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="astra_reversal/.deps/astra-dev-proposal-replay-input"
    )
    parser.add_argument("--verify-inputs-only", action="store_true")
    parser.add_argument("--prepare-inputs", metavar="SOURCE_RUN")
    parser.add_argument(
        "--plan", default="astra_reversal/.deps/astra-dev-proposal-replay-plan.json"
    )
    parser.add_argument("--archive-catalog")
    args = parser.parse_args()
    if args.prepare_inputs:
        require(
            args.archive_catalog, "Preparing inputs requires a private archive catalog"
        )
        value = prepare_inputs(
            args.prepare_inputs, args.plan, args.archive_catalog, args.input
        )
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "bundle_sha256": value["bundle_sha256"],
                    "source_archive": value["source_archive"],
                }
            ),
            flush=True,
        )
        return 0
    if args.verify_inputs_only:
        value = load_inputs(args.input)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "cases": len(value["cases"]),
                    "bundle_sha256": value["provenance"]["bundle_sha256"],
                }
            ),
            flush=True,
        )
        return 0
    return run_probe(args.input)


if __name__ == "__main__":
    raise SystemExit(main())
