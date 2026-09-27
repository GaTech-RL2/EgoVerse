"""Train once, then evaluate the frozen recipe without an inference credential."""

import json
import os
import platform
import tarfile
import time
from pathlib import Path

import numpy as np

from astra_reversal.action_adapter import ActionAdapter, ActionSpec
from astra_reversal.config import BenchmarkConfig
from astra_reversal.intervention_rollout import capture_reset_manifest
from astra_reversal.intervention_search import write_json
from astra_reversal.libero_runner import configure_libero
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.frs_policy_improvement import frozen_parameter_receipt
from astra_reversal.osmo.interpolation import (
    load_banks,
    load_frozen_policy,
    native_preflight,
)
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.recipe_corpus import iter_samples, load_corpus
from astra_reversal.recipe_experiment import (
    assignment,
    load_protocol,
    recorded_schedules,
    rollout,
)
from astra_reversal.recipe_learning import (
    ResidualActionHead,
    base_identity,
    capture_samples,
)
from astra_reversal.recipe_selector import PhaseSelector, prepare_native_features
from astra_reversal.records import digest, file_sha256


def environment_root(suite):
    return ROOT / (
        "astra_reversal/.deps/libero"
        if suite == "libero_10"
        else "astra_reversal/.deps/libero-ood/third_party/modified_libero"
    )


def _seal(directory):
    return {
        str(p.relative_to(directory)): {
            "bytes": p.stat().st_size,
            "sha256": file_sha256(p),
        }
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _checkpoint_bundle(archive, bundle):
    destination = RESULTS / "learned_bundle.tar.gz"
    with tarfile.open(destination, "w:gz", compresslevel=1) as stream:
        stream.add(bundle, arcname="bundle")
    metadata = {
        "schema_version": "recipe-bundle-1.0",
        "key": archive.prefix + "/learned_bundle.tar.gz",
        "sha256": file_sha256(destination),
        "bytes": destination.stat().st_size,
    }
    archive.client.upload_file(str(destination), "rldb", metadata["key"])
    write_json(RESULTS / "learned_bundle.json", metadata)
    return metadata


def _load_bundle(archive):
    workflow = os.environ["ASTRA_RECIPE_TRAINING_WORKFLOW"]
    if not workflow.startswith("astra-pi05-recipe-training-") or "/" in workflow:
        raise ValueError("Unexpected recipe training workflow")
    expected = os.environ["ASTRA_RECIPE_BUNDLE_SHA256"]
    destination = ROOT / "learned_bundle.tar.gz"
    key = (
        f"experiments/astra-reversal-20260924/{workflow}/worker_0/learned_bundle.tar.gz"
    )
    archive.client.download_file("rldb", key, str(destination))
    if file_sha256(destination) != expected:
        raise ValueError("Learned checkpoint archive bytes differ")
    output = ROOT / "learned_recipe_models"
    output.mkdir(exist_ok=False)
    with tarfile.open(destination) as stream:
        for member in stream.getmembers():
            if not member.name.startswith("bundle/") and member.name != "bundle":
                raise ValueError("Unexpected learned bundle member")
            if member.issym() or member.islnk() or ".." in Path(member.name).parts:
                raise ValueError("Unsafe learned bundle member")
        stream.extractall(output, filter="data")
    bundle = output / "bundle"
    seal = json.loads((bundle / "seal.json").read_text())
    for name, expected_file in seal.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe learned bundle seal")
        path = bundle / name
        if (
            file_sha256(path) != expected_file["sha256"]
            or path.stat().st_size != expected_file["bytes"]
        ):
            raise ValueError("Learned bundle file differs")
    return bundle


def train(archive, protocol):
    corpus_path = ROOT / "astra_reversal/.deps/recipe-inputs/corpus"
    metadata, _ = load_corpus(corpus_path)
    samples = list(iter_samples(corpus_path))
    specs = {r["trajectory_id"]: r["action_spec"] for r in metadata["trajectories"]}
    if len(samples) != 413 or len(metadata["trajectories"]) != 19:
        raise ValueError("Historical training denominator differs")
    for sample in samples:
        sample["action_spec"] = specs[sample["trajectory_id"]]
    write_json(RESULTS / "corpus_metadata.json", metadata)
    schedules = recorded_schedules(samples)
    benchmark = BenchmarkConfig.preset("libero_10")
    manifest = capture_reset_manifest(
        environment_root("libero_10"),
        benchmark,
        seed=61,
        cases=[(i, 0) for i in range(4)],
        output=RESULTS / "training_reset_manifest.json",
        split="development",
    )
    _, create = configure_libero(
        environment_root("libero_10"), benchmark, RESULTS / "libero_config"
    )
    policy = load_frozen_policy()
    before = frozen_parameter_receipt(policy)
    write_json(RESULTS / "frozen_weights_before.json", before)
    anchor_rows = []
    for index, entry in enumerate(manifest["episodes"]):
        write_json(
            RESULTS / "progress.json",
            {"status": "collecting_native_anchors", "complete": index, "total": 4},
        )
        archive.sync()
        result, new_samples = rollout(
            policy,
            create,
            entry,
            "native",
            RESULTS / "anchor_rollouts" / f"task_{entry['task_id']}",
            collect=True,
            probe=index == 0,
        )
        anchor_rows.append(result)
        samples.extend(new_samples)
        write_json(RESULTS / "anchor_collection.json", anchor_rows)
        archive.sync()
    admission = {
        "schema_version": "recipe-admission-1.0",
        "historical_metadata_sha256": file_sha256(corpus_path / "metadata.json"),
        "historical_array_sha256": file_sha256(corpus_path / "arrays.npz"),
        "protocol_sha256": digest(protocol),
        "training_reset_sha256": manifest["sha256"],
        "fresh_anchor_rollouts": [
            {
                "episode_id": row["episode_id"],
                "success": row["success"],
                "actions": row["actions_executed"],
                "admitted": row["success"],
            }
            for row in anchor_rows
        ],
        "samples": [
            {
                "sample_id": r["sample_id"],
                "kind": r["kind"],
                "trajectory_id": r["trajectory_id"],
                "executed_actions": len(r["executed_actions"]),
                "source_receipt_sha256": r["source_receipt_sha256"],
            }
            for r in samples
        ],
        "outcome_admission": "successful completed trajectory; no pairwise judge",
        "provider_calls": 0,
        "provider_tokens": 0,
    }
    write_json(RESULTS / "admission.json", admission)
    features, choices, trajectory_ids, executed, anchors, feature_receipts = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    began = time.perf_counter()
    for index, sample in enumerate(samples):
        observation, prompt = sample["observation"], sample["original_prompt"]
        condition, vector, feature_receipt = prepare_native_features(
            policy, observation, prompt
        )
        del condition
        features.append(vector)
        choices.append(sample["choice"])
        trajectory_ids.append(sample["trajectory_id"])
        spec = ActionSpec(**sample["action_spec"])
        adapter = ActionAdapter(spec, policy.input_transform, policy.output_transform)
        bank = capture_samples(
            policy,
            adapter,
            observation,
            prompt,
            sample["executed_actions"],
            sample["sample_id"],
            sample_id=sample["sample_id"],
            source_receipt_sha256=sample["source_receipt_sha256"],
            anchor=sample["kind"] == "anchor",
        )
        bank.save(RESULTS / "arrays" / "feature_banks" / sample["sample_id"])
        (anchors if sample["kind"] == "anchor" else executed).append(bank)
        feature_receipts.append(
            {
                "sample_id": sample["sample_id"],
                "selector": feature_receipt,
                "flow_bank": bank.metadata(),
            }
        )
        if index % 25 == 0 or index + 1 == len(samples):
            write_json(
                RESULTS / "progress.json",
                {
                    "status": "extracting_frozen_features",
                    "complete": index + 1,
                    "total": len(samples),
                    "seconds": time.perf_counter() - began,
                },
            )
            archive.sync()
    matrix = np.stack(features)
    np.savez_compressed(RESULTS / "arrays" / "selector_features.npz", features=matrix)
    write_json(RESULTS / "feature_receipts.json", feature_receipts)
    extraction_seconds = time.perf_counter() - began
    admission_sha = file_sha256(RESULTS / "admission.json")
    write_json(
        RESULTS / "progress.json",
        {"status": "training_selector_and_head", "samples": len(samples)},
    )
    archive.sync()
    selector = PhaseSelector.for_choices(matrix, choices, device=policy.device)
    selector_receipt = selector.fit(
        matrix, choices, trajectory_ids, source_sha256=admission_sha
    )
    head = ResidualActionHead(base_identity(policy), device=policy.device)
    head_receipt = head.fit(executed, anchors, admission_receipt_sha256=admission_sha)
    after = frozen_parameter_receipt(policy)
    write_json(RESULTS / "frozen_weights_after.json", after)
    if (
        before != after
        or head_receipt["optimizer_steps"] != 1000
        or selector_receipt["optimizer_steps"] != 1000
    ):
        raise RuntimeError("Recipe training parameter/update gate failed")
    bundle = RESULTS / "bundle"
    bundle.mkdir()
    selector.save(bundle / "selector")
    head.save(bundle / "head")
    write_json(bundle / "schedules.json", schedules)
    write_json(bundle / "protocol.json", protocol)
    receipt = {
        "schema_version": "recipe-training-1.0",
        "status": "complete",
        "workflow": os.environ["ASTRA_RUN_ID"],
        "protocol_sha256": digest(protocol),
        "admission_sha256": admission_sha,
        "selector": selector_receipt,
        "head": head_receipt,
        "base_identity": base_identity(policy),
        "base_weights_unchanged": before == after,
        "native_parameter_sha256": after["sha256"],
        "historical_windows": 413,
        "training_windows": len(samples),
        "corrected_trajectories": 12,
        "fresh_native_anchor_rollouts": anchor_rows,
        "feature_extraction_seconds": extraction_seconds,
        "feature_velocity_evaluations": sum(
            b.counts["velocity_evaluations"] for b in [*executed, *anchors]
        ),
        "feature_prefix_evaluations": len(samples)
        + sum(b.counts["prefix_preparations"] for b in [*executed, *anchors]),
        "provider_calls": 0,
        "provider_tokens": 0,
        "new_provider_cost_scope": "excludes historical teacher acquisition",
    }
    write_json(bundle / "training_receipt.json", receipt)
    write_json(bundle / "seal.json", _seal(bundle))
    artifact = _checkpoint_bundle(archive, bundle)
    write_json(RESULTS / "training_receipt.json", receipt)
    write_json(
        RESULTS / "progress.json",
        {"status": "complete", "phase": "training", "learned_bundle": artifact},
    )


def evaluate(archive, protocol, worker):
    import torch

    bundle = _load_bundle(archive)
    if json.loads((bundle / "protocol.json").read_text()) != protocol:
        raise ValueError("Frozen training and evaluation protocols differ")
    receipt = json.loads((bundle / "training_receipt.json").read_text())
    if (
        receipt["status"] != "complete"
        or not receipt["base_weights_unchanged"]
        or receipt["provider_calls"]
    ):
        raise ValueError("Training gates did not pass")
    write_json(RESULTS / "training_receipt.json", receipt)
    suite, tasks = assignment(worker)
    benchmark = BenchmarkConfig.preset(suite)
    manifest = capture_reset_manifest(
        environment_root(suite),
        benchmark,
        seed=61,
        cases=[(i, j) for i in tasks for j in (1, 2)],
        output=RESULTS / "evaluation_reset_manifest.json",
    )
    _, create = configure_libero(
        environment_root(suite), benchmark, RESULTS / "libero_config"
    )
    banks = load_banks(
        archive, ROOT / "astra_reversal/.deps/interpolation-inputs/bank_inventory.json"
    )
    policy = load_frozen_policy()
    before = frozen_parameter_receipt(policy)
    write_json(RESULTS / "frozen_weights_before.json", before)
    selector = PhaseSelector.load(bundle / "selector", device=policy.device)
    if (
        before["sha256"] != receipt["native_parameter_sha256"]
        or base_identity(policy) != receipt["base_identity"]
    ):
        raise ValueError("Selector/head training native model identity differs")
    head = ResidualActionHead.load(
        bundle / "head",
        expected_base_identity=base_identity(policy),
        device=policy.device,
    )
    head.eval().requires_grad_(False)
    if (
        selector.training["test_only"]
        or selector.training["optimizer_steps"] != 1000
        or head.optimizer_steps != 1000
    ):
        raise ValueError("Test-only or incomplete model cannot be evaluated")
    head_sha = head.parameter_sha256()
    schedules = json.loads((bundle / "schedules.json").read_text())
    rows = []
    write_json(
        RESULTS / "frozen_plan.json",
        {
            "protocol_sha256": digest(protocol),
            "reset_manifest_sha256": manifest["sha256"],
            "training_bundle_sha256": os.environ["ASTRA_RECIPE_BUNDLE_SHA256"],
            "episode_ids": [e["episode_id"] for e in manifest["episodes"]],
            "arms": protocol["arms"],
            "suite": suite,
            "worker": worker,
        },
    )
    for index, entry in enumerate(manifest["episodes"]):
        expected = None
        for arm in protocol["arms"]:
            write_json(
                RESULTS / "progress.json",
                {
                    "status": "evaluating",
                    "completed_rollouts": len(rows),
                    "assigned_rollouts": len(manifest["episodes"])
                    * len(protocol["arms"]),
                    "episode_id": entry["episode_id"],
                    "arm": arm,
                },
            )
            archive.sync()
            directory = (
                RESULTS
                / "rollouts"
                / f"task_{entry['task_id']}_state_{entry['initial_state_id']}"
                / arm
            )
            with torch.no_grad():
                result, _ = rollout(
                    policy,
                    create,
                    entry,
                    arm,
                    directory,
                    selector=selector,
                    head=head,
                    banks=banks,
                    schedules=schedules,
                    expected_reset=expected,
                    probe=index == 0 and arm == "native",
                )
            expected = result["reset_audit"]
            result["relative_directory"] = str(directory.relative_to(RESULTS))
            result["summary_sha256"] = file_sha256(directory / "summary.json")
            rows.append(result)
            write_json(RESULTS / "rollouts.json", rows)
            archive.sync()
    after = frozen_parameter_receipt(policy)
    write_json(RESULTS / "frozen_weights_after.json", after)
    if before != after or head.parameter_sha256() != head_sha:
        raise RuntimeError("Evaluation changed frozen native/learned parameters")
    write_json(
        RESULTS / "progress.json",
        {
            "status": "complete",
            "phase": "evaluation",
            "physical_rollouts": len(rows),
            "worker": worker,
        },
    )
    write_json(
        RESULTS / "completion_receipt.json",
        {
            "schema_version": "recipe-evaluation-worker-1.0",
            "status": "complete",
            "worker": worker,
            "suite": suite,
            "episodes": len(manifest["episodes"]),
            "physical_rollouts": len(rows),
            "native_parameter_sha256": after["sha256"],
            "head_parameter_sha256": head_sha,
            "training_bundle_sha256": os.environ["ASTRA_RECIPE_BUNDLE_SHA256"],
            "files": _seal(RESULTS),
            "provider_calls": 0,
            "provider_tokens": 0,
        },
    )


def main():
    import torch

    os.environ.pop("NVIDIA_INFERENCE_API_KEY", None)
    phase, worker = (
        os.environ["ASTRA_RECIPE_PHASE"],
        int(os.environ["ASTRA_WORKER_INDEX"]),
    )
    if phase not in ("training", "evaluation") or (phase == "training" and worker != 0):
        raise ValueError("Invalid learned recipe assignment")
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("One allocated OSMO L40S is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(worker)
    protocol = load_protocol()
    write_json(RESULTS / "protocol.json", protocol)
    write_json(
        RESULTS / "runtime.json",
        {
            "phase": phase,
            "worker": worker,
            "workflow": os.environ["ASTRA_RUN_ID"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "provider_credential_attached": False,
            "tf32": False,
            "started_unix": time.time(),
        },
    )
    write_json(RESULTS / "progress.json", {"status": "preflight", "phase": phase})
    archive.sync()
    try:
        native_preflight(archive)
        if phase == "training":
            train(archive, protocol)
        else:
            evaluate(archive, protocol, worker)
    except BaseException as exc:
        write_json(
            RESULTS / "failure.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
