"""Synthetic recorded-data contracts; no policy, simulator or provider calls."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from astra_reversal import recipe_report as report
from astra_reversal.config import BenchmarkConfig
from astra_reversal.recipe_learning import training_config
from astra_reversal.recipe_selector import selector_config
from astra_reversal.records import digest, file_sha256

H = "a" * 64
HEAD = "b" * 64
CHOICE = {"operator": "tei", "source_a_id": "10", "source_b_id": "14", "alpha": 0.25}
NATIVE = {"operator": "native"}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def inventory(root):
    return {
        str(p.relative_to(root)): {"sha256": file_sha256(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def make_manifest(suite, tasks, states):
    entries = []
    for task in tasks:
        for state in states:
            initial = np.array([task, state], np.float64)
            model = {
                "body_pos": np.array([[task, state, 0.0]], np.float64),
                "body_quat": np.array([[1, 0, 0, 0]], np.float64),
            }
            entry = {
                "episode_id": report._id(suite, task, state),
                "suite": suite,
                "task_id": task,
                "initial_state_id": state,
                "seed": 61,
                "instruction": f"{suite} task {task}",
                "reset_state": initial.tolist(),
                "reset_state_sha256": digest(initial),
                "reset_model": {k: v.tolist() for k, v in model.items()},
                "reset_model_sha256": digest(model),
                "bddl_sha256": H,
                "initially_successful": False,
                "prescribed_state_asset_sha256": H if suite == "libero_10" else None,
            }
            entries.append(entry)
    value = {
        "schema_version": "intervention_reset_v1",
        "seed": 61,
        "benchmark": asdict(BenchmarkConfig.preset(suite)),
        "cases": [[t, s] for t in tasks for s in states],
        "episodes": entries,
    }
    value["sha256"] = digest(value)
    return value


def runtime(phase, worker):
    return {
        "phase": phase,
        "worker": worker,
        "workflow": "synthetic-" + phase,
        "payload_sha256": H,
        "gpu": "NVIDIA L40S",
        "provider_credential_attached": False,
        "tf32": False,
    }


def weights(root):
    tensors = {"frozen": {"shape": [1], "dtype": "torch.float32", "sha256": H}}
    value = {"tensors": tensors, "sha256": digest(tensors)}
    write(root / "frozen_weights_before.json", value)
    write(root / "frozen_weights_after.json", value)
    return value["sha256"]


def physical(
    root, relative, entry, arm, success, *, probe=False, actions=5, schedules=None
):
    directory = root / relative
    directory.mkdir(parents=True)
    (directory / "arrays").mkdir()
    (directory / "rollout.mp4").write_bytes(b"synthetic-test-video-not-exported")
    reset = {
        k: entry[k]
        for k in (
            "episode_id",
            "seed",
            "reset_state_sha256",
            "reset_model_sha256",
            "bddl_sha256",
        )
    }
    reset.update(
        post_stabilization_state_sha256=H,
        post_stabilization_model_sha256=entry["reset_model_sha256"],
        post_stabilization_observation_sha256=H,
        stabilization_steps=10,
        initial_success=False,
        initial_terminated=False,
    )
    reset["sha256"] = digest(reset)
    summary = {
        k: entry[k]
        for k in (
            "episode_id",
            "suite",
            "task_id",
            "seed",
            "initial_state_id",
            "instruction",
        )
    }
    summary.update(
        arm=arm,
        status="complete",
        success=success,
        actions_executed=actions,
        policy_replans=(actions + 4) // 5,
        wall_seconds=20.0,
        reset_seconds=1.0,
        policy_seconds=10.0,
        environment_seconds=5.0,
        initial_success=False,
        captured_initial_success=False,
        zero_action_success=False,
        terminated=not success,
        action_budget=520 if entry["suite"] == "libero_10" else 300,
        execute_steps=5,
        reset_audit=reset,
        video_path="/private/remote/rollout.mp4",
        video_sha256=file_sha256(directory / "rollout.mp4"),
        video_fps=20,
        provider_calls=0,
        provider_tokens=0,
        probe_velocity_evaluations=30 if probe else 0,
        probe_prefix_evaluations=3 if probe else 0,
        clipped_predicted_values=0,
    )
    summary["checks"] = (
        {
            key: {"max_abs": 0.0}
            for key in (
                "upstream_native_parity",
                "feature_capture_parity",
                "zero_head_parity",
            )
        }
        if probe
        else {}
    )
    events = [{"kind": "recipe_rollout_start", "entry": entry, "arm": arm}]
    if probe:
        events.append({"kind": "recipe_native_probes", "checks": summary["checks"]})
    decisions, generations, prefixes = [], [], 0
    choice, gate = NATIVE, False
    # Descriptors intentionally represent synthetic metadata; the mock archive
    # receipt explicitly does not claim an independent array replay.
    for step in range(0, actions, 5):
        if step % 25 == 0 and arm in (
            "recorded_schedule",
            "learned_selector",
            "gated_flow_head",
        ):
            if arm == "recorded_schedule":
                schedule = (schedules or {}).get(entry["instruction"])
                choice = schedule["choices"][0]["choice"] if schedule else NATIVE
                gate = choice != NATIVE
                decision = {
                    "step": step,
                    "choice": choice,
                    "gate_active": gate,
                    "gate_probability": None,
                }
            else:
                choice, gate = CHOICE, True
                decision = {
                    "step": step,
                    "choice": choice,
                    "gate_active": gate,
                    "gate_probability": 0.75,
                    "feature_receipt": {"feature_sha256": H},
                }
                events.append(
                    {
                        "kind": "recipe_selector_decision",
                        "step": step,
                        "decision": decision,
                    }
                )
            decisions.append(decision)
        head = arm == "flow_head" or (arm == "gated_flow_head" and gate)
        edited = head or (arm == "learned_selector" and choice != NATIVE)
        feature_step = arm in ("learned_selector", "gated_flow_head") and step % 25 == 0
        prefixes += 1 + int(feature_step and edited)
        identity = int(digest(entry["episode_id"])[:16], 16)
        noise = (
            np.random.default_rng(np.random.SeedSequence([61, identity, 0, step, 0]))
            .standard_normal((1, 10, 32))
            .astype(np.float32)
        )
        name = f"arrays/noise_{step}.npy"
        np.save(directory / name, noise)
        descriptor = {
            "array": name,
            "shape": [1, 10, 32],
            "dtype": "float32",
            "sha256": digest(noise),
        }
        record = {
            "step": step,
            "choice": choice,
            "gate_active": gate,
            "head_active": head,
            "noise_sha256": digest(noise),
        }
        generations.append(record)
        provenance = (
            {
                "operator": "bounded_final_action_expert_residual",
                "native_prefix_unchanged": True,
                "original_condition_id": H,
                "original_prompt_sha256": digest(entry["instruction"]),
                "direct_rows_5_to_9_unchanged": True,
                "direct_padding_unchanged": True,
                "hard_clipping_used": False,
                "head": {
                    "parameter_sha256": HEAD,
                    "optimizer_steps": 1000,
                    "test_only": False,
                },
                "residual_max_abs": 0.2,
            }
            if head
            else None
        )
        events.append(
            {
                "kind": "recipe_generation",
                "observation_step": step,
                "observation": {
                    k: {}
                    for k in (
                        "observation/image",
                        "observation/wrist_image",
                        "observation/state",
                    )
                },
                "noise": descriptor,
                "condition_id": H,
                "choice": choice,
                "gate_active": gate,
                "head_active": head,
                "provenance": provenance,
                "clipping": {"count": 0},
                "velocity_evaluations": 10,
            }
        )
    summary.update(
        decisions=decisions,
        generations=generations,
        velocity_evaluations=len(generations) * 10,
        prefix_evaluations=prefixes,
        head_active_actions=actions if arm in ("flow_head", "gated_flow_head") else 0,
        gate_active_actions=actions if gate else 0,
    )
    events.append({"kind": "recipe_rollout_complete", "result": summary})
    (directory / "events.jsonl").write_text(
        "".join(
            json.dumps({**e, "sequence": i, "timestamp": float(i)}) + "\n"
            for i, e in enumerate(events)
        )
    )
    write(directory / "summary.json", summary)
    return summary


def corpus():
    sources, trajectories, windows = [], [], []
    prompts = [f"libero_goal_ood task {i}" for i in range(4)] + [
        f"libero_spatial_ood task {i}" for i in range(4)
    ]
    for trajectory in range(19):
        teacher = trajectory < 12
        length = ([21] * 11 + [24] + [23] * 4 + [22] * 3)[trajectory]
        source_id, trajectory_id = f"source{trajectory}", digest(trajectory)
        source = {"source_id": source_id, "feedback_audit_sha256": digest(source_id)}
        sources.append(source)
        prompt = prompts[trajectory % 8] if teacher else f"anchor task {trajectory}"
        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "sample_kind": "astra_success" if teacher else "native_anchor",
                "success": True,
            }
        )
        for step in range(length):
            windows.append(
                {
                    "sample_id": digest((trajectory, step)),
                    "source_id": source_id,
                    "trajectory_id": trajectory_id,
                    "episode_id": f"training:task{trajectory}",
                    "sample_kind": "astra_success" if teacher else "native_anchor",
                    "executed_count": (
                        3 if trajectory < 9 else 2 if teacher or trajectory < 14 else 3
                    )
                    if step == length - 1
                    else 5,
                    "instruction": prompt,
                    "action_start": step * 5,
                    "choice": CHOICE if teacher else NATIVE,
                    "source_study": "phase",
                    "arm": "astra_tei" if teacher else "recovered_noise",
                    "revision": 1,
                }
            )
    value = {
        "status": "complete",
        "sources": sources,
        "trajectories": trajectories,
        "windows": windows,
        "trajectory_count": 19,
        "window_count": 413,
        "executed_action_count": 2022,
        "arrays": {"sha256": H},
    }
    value["corpus_id"] = digest(value)
    return value


def make_training(root, protocol, *, successful_fresh=False):
    root.mkdir()
    write(root / "protocol.json", protocol)
    write(root / "runtime.json", runtime("training", 0))
    base = {
        "native": {"artifact_sha256": {"weights": H}},
        "horizon": 10,
        "model_dim": 32,
        "hidden_dim": 1024,
    }
    write(root / "checkpoint.json", base["native"])
    native_sha = weights(root)
    manifest = make_manifest("libero_10", range(4), [0])
    write(root / "training_reset_manifest.json", manifest)
    anchors = [
        physical(
            root,
            f"anchor_rollouts/task_{i}",
            entry,
            "native",
            successful_fresh and i == 0,
            probe=i == 0,
            actions=7 if successful_fresh and i == 0 else 5,
        )
        for i, entry in enumerate(manifest["episodes"])
    ]
    write(root / "anchor_collection.json", anchors)
    historical = corpus()
    write(root / "corpus_metadata.json", historical)
    sources = {s["source_id"]: s for s in historical["sources"]}
    samples = [
        {
            "sample_id": w["sample_id"],
            "trajectory_id": w["trajectory_id"],
            "kind": "anchor" if w["sample_kind"] == "native_anchor" else "correction",
            "executed_actions": w["executed_count"],
            "source_receipt_sha256": sources[w["source_id"]]["feedback_audit_sha256"],
        }
        for w in historical["windows"]
    ]
    for anchor in anchors:
        if anchor["success"]:
            checksum = file_sha256(
                root / f"anchor_rollouts/task_{anchor['task_id']}/summary.json"
            )
            for step in range(0, anchor["actions_executed"], 5):
                samples.append(
                    {
                        "sample_id": digest((anchor["episode_id"], step, checksum)),
                        "trajectory_id": digest(anchor["episode_id"]),
                        "kind": "anchor",
                        "executed_actions": min(5, anchor["actions_executed"] - step),
                        "source_receipt_sha256": checksum,
                    }
                )
    admission = {
        "historical_metadata_sha256": file_sha256(root / "corpus_metadata.json"),
        "historical_array_sha256": H,
        "protocol_sha256": digest(protocol),
        "training_reset_sha256": manifest["sha256"],
        "fresh_anchor_rollouts": [
            {
                "episode_id": r["episode_id"],
                "success": r["success"],
                "actions": r["actions_executed"],
                "admitted": r["success"],
            }
            for r in anchors
        ],
        "samples": samples,
        "provider_calls": 0,
        "provider_tokens": 0,
    }
    write(root / "admission.json", admission)
    features, bank_ids = [], {"executed": [], "anchor": []}
    for sample in samples:
        kind = "anchor" if sample["kind"] == "anchor" else "executed"
        bank = {
            "provenance": {
                "base_identity": base,
                "test_only": False,
                "base_parameter_versions_unchanged": True,
                "kind": kind,
                "window": {
                    "source_id": sample["sample_id"],
                    "source_receipt_sha256": sample["source_receipt_sha256"],
                    "observation_sha256": H,
                    "condition_id": H,
                },
                "counts": {
                    "velocity_evaluations": 18,
                    "prefix_preparations": 2,
                    "feature_rows": 40
                    if kind == "anchor"
                    else 8 * sample["executed_actions"],
                },
            },
            "arrays": {},
        }
        bank["bank_id"] = digest(bank)
        bank_ids[kind].append(bank["bank_id"])
        features.append(
            {
                "sample_id": sample["sample_id"],
                "flow_bank": bank,
                "selector": {
                    "prefix_evaluations": 1,
                    "velocity_evaluations": 0,
                    "raw_observation_sha256": H,
                    "condition_id": H,
                },
            }
        )
    write(root / "feature_receipts.json", features)
    selector = {
        "source_sha256": file_sha256(root / "admission.json"),
        "optimizer_steps": 1000,
        "test_only": False,
        "samples": len(samples),
        "config": selector_config(),
        "wall_seconds": 2.0,
        "loss_trace": [
            {
                "step": step,
                "loss": 1.0,
                "gate_bce": 0.5,
                "pair_ce": 0.3,
                "alpha_mse": 0.2,
            }
            for step in [*range(1, 1000, 50), 1000]
        ],
    }
    head = {
        "admission_receipt_sha256": selector["source_sha256"],
        "optimizer_steps": 1000,
        "total_optimizer_steps": 1000,
        "test_only": False,
        "status": "fitted",
        "trainable_parameters": 7175,
        "velocity_evaluations": 0,
        "base_backward_passes": 0,
        "before": {
            "masked_residual_mse": 1.0,
            "anchor_mse": 0.0,
            "hard_clipping_used": False,
        },
        "after": {
            "masked_residual_mse": 0.5,
            "anchor_mse": 0.1,
            "hard_clipping_used": False,
        },
        "executed_bank_ids": bank_ids["executed"],
        "anchor_bank_ids": bank_ids["anchor"],
        "feature_rows": 1248 * 8,
        "anchor_rows": len(bank_ids["anchor"]) * 40,
        "after_parameter_sha256": HEAD,
        "wall_seconds": 3.0,
        "device": "cuda:0",
    }
    receipt = {
        "schema_version": "recipe-training-1.0",
        "status": "complete",
        "workflow": "synthetic-training",
        "protocol_sha256": digest(protocol),
        "admission_sha256": selector["source_sha256"],
        "selector": selector,
        "head": head,
        "base_identity": base,
        "base_weights_unchanged": True,
        "native_parameter_sha256": native_sha,
        "historical_windows": 413,
        "training_windows": len(samples),
        "corrected_trajectories": 12,
        "fresh_native_anchor_rollouts": anchors,
        "feature_extraction_seconds": 4.0,
        "feature_velocity_evaluations": len(samples) * 18,
        "feature_prefix_evaluations": len(samples) * 3,
        "provider_calls": 0,
        "provider_tokens": 0,
    }
    write(root / "training_receipt.json", receipt)
    bundle = root / "bundle"
    write(bundle / "training_receipt.json", receipt)
    write(bundle / "protocol.json", protocol)
    schedules = {}
    for t in historical["trajectories"][:12]:
        rows = [
            w for w in historical["windows"] if w["trajectory_id"] == t["trajectory_id"]
        ]
        first = rows[0]
        candidate = {
            "trajectory_id": t["trajectory_id"],
            "source_receipt_sha256": sources[first["source_id"]][
                "feedback_audit_sha256"
            ],
            "choices": [{"step": w["action_start"], "choice": CHOICE} for w in rows],
        }
        old = schedules.get(first["instruction"])
        if old is None or first["episode_id"] < next(
            w["episode_id"]
            for w in historical["windows"]
            if w["trajectory_id"] == old["trajectory_id"]
        ):
            schedules[first["instruction"]] = candidate
    write(bundle / "schedules.json", schedules)
    (bundle / "selector").mkdir()
    (bundle / "head").mkdir()
    for path in (
        bundle / "selector/weights.pt",
        bundle / "selector/normalization.npz",
        bundle / "head/state.pt",
    ):
        path.write_bytes(b"synthetic small checkpoint; never loaded")
    source = Path(report.__file__).parent
    write(
        bundle / "selector/metadata.json",
        {
            "training": selector,
            "implementation_sha256": file_sha256(source / "recipe_selector.py"),
            "files": {
                name: file_sha256(bundle / "selector" / name)
                for name in ("weights.pt", "normalization.npz")
            },
        },
    )
    write(
        bundle / "head/manifest.json",
        {
            "history": [head],
            "base_identity": base,
            "parameter_sha256": HEAD,
            "optimizer_steps": 1000,
            "test_only": False,
            "source_sha256": file_sha256(source / "recipe_learning.py"),
            "config": training_config(),
            "state_file_sha256": file_sha256(bundle / "head/state.pt"),
        },
    )
    write(bundle / "seal.json", inventory(bundle))
    write(root / "learned_bundle.json", {"sha256": H, "bytes": 100})
    return receipt, schedules


def make_worker(root, worker, protocol, receipt, schedules):
    root.mkdir()
    write(root / "runtime.json", runtime("evaluation", worker))
    write(root / "protocol.json", protocol)
    write(root / "training_receipt.json", receipt)
    write(root / "checkpoint.json", receipt["base_identity"]["native"])
    native = weights(root)
    suite, tasks = report._assignment(worker)
    manifest = make_manifest(suite, tasks, [1, 2])
    write(root / "evaluation_reset_manifest.json", manifest)
    write(root / "bank_inventory.json", {"bank": H})
    write(
        root / "frozen_plan.json",
        {
            "protocol_sha256": digest(protocol),
            "reset_manifest_sha256": manifest["sha256"],
            "training_bundle_sha256": H,
            "episode_ids": [e["episode_id"] for e in manifest["episodes"]],
            "arms": list(report.ARMS),
            "suite": suite,
            "worker": worker,
        },
    )
    rows = []
    for entry in manifest["episodes"]:
        for arm in report.ARMS:
            success = (
                entry["task_id"] % 2 == 0
                if arm in ("native", "recorded_schedule")
                else entry["task_id"] % 3 == 0
            )
            relative = f"rollouts/task_{entry['task_id']}_state_{entry['initial_state_id']}/{arm}"
            row = physical(
                root, relative, entry, arm, success, probe=not rows, schedules=schedules
            )
            rows.append(
                {
                    **row,
                    "relative_directory": relative,
                    "summary_sha256": file_sha256(root / relative / "summary.json"),
                }
            )
    write(root / "rollouts.json", rows)
    write(root / "progress.json", {"status": "complete"})
    seal = {
        "schema_version": "recipe-evaluation-worker-1.0",
        "status": "complete",
        "worker": worker,
        "suite": suite,
        "episodes": len(manifest["episodes"]),
        "physical_rollouts": len(rows),
        "native_parameter_sha256": native,
        "head_parameter_sha256": HEAD,
        "training_bundle_sha256": H,
        "provider_calls": 0,
        "provider_tokens": 0,
        "files": inventory(root),
    }
    write(root / "completion_receipt.json", seal)


def archive_receipt(root, phase, worker):
    files = inventory(root)
    return {
        "schema_version": "recipe-archive-audit-1.0",
        "status": "passed",
        "complete": True,
        "phase": phase,
        "worker": worker,
        "workflow": "synthetic-" + phase,
        "archive": {"sha256": H, "bytes": 1000, "etag": "synthetic"},
        "files": files,
        "member_inventory": {
            "sha256": H,
            "members": len(files),
            "bytes": sum(r["bytes"] for r in files.values()),
        },
        "checks": {
            "full_archive_stream_verified": True,
            "all_member_bytes_verified": True,
            "training_bundle_seal_verified": phase == "training",
            "completion_seal_verified": phase == "evaluation",
            "array_descriptors_verified": False,
        },
        "source_sha256": H,
    }


@pytest.fixture(scope="module")
def complete_fixture(tmp_path_factory):
    base = tmp_path_factory.mktemp("synthetic_recipe")
    protocol = json.loads(
        (
            Path(report.__file__).parent / "configs/learned_correction_recipe_v1.json"
        ).read_text()
    )
    training = base / "training"
    receipt, schedules = make_training(training, protocol)
    workers = []
    for worker in range(5):
        root = base / f"worker_{worker}"
        make_worker(root, worker, protocol, receipt, schedules)
        workers.append(root)
    audits = {
        "training": archive_receipt(training, "training", 0),
        **{
            f"worker_{i}": archive_receipt(p, "evaluation", i)
            for i, p in enumerate(workers)
        },
    }
    return training, workers, audits


def test_complete_cartesian_pairing_cost_and_harm(complete_fixture):
    training, workers, receipts = complete_fixture
    result = report.build_report(
        training, workers, archive_receipts=receipts, require_complete=True
    )
    assert result["efficacy_released"] and len(result["rows"]) == 240
    assert result["groups"]["ood"]["methods"]["native"]["cases"] == 40
    assert result["groups"]["id_panel"]["methods"]["native"]["cases"] == 8
    pair = result["groups"]["ood"]["paired"][1]
    assert (
        pair["both_success"],
        pair["method_only_success"],
        pair["reference_only_success"],
        pair["both_failure"],
    ) == (8, 8, 12, 12)
    assert result["cost"]["evaluation"]["total_velocity_evaluations"] == 2400 + 150
    assert result["cost"]["all_new_velocity_evaluations"] == 2550 + 7434 + 70
    assert result["cost"]["new_provider_calls"] == 0
    assert result["validation"]["array_contents_replayed"] is False
    assert "private/remote" not in json.dumps(result)


def test_missing_worker_or_archive_withholds_efficacy(complete_fixture):
    training, workers, receipts = complete_fixture
    partial = report.build_report(
        training,
        workers[:4],
        archive_receipts={k: v for k, v in receipts.items() if k != "worker_4"},
    )
    assert (
        partial["status"] == "incomplete"
        and partial["groups"] == {}
        and partial["per_task"] == []
    )
    assert partial["coverage"]["verified_physical_rollouts"] == 200
    assert partial["cost"]["evaluation_is_lower_bound"] is True
    without = report.build_report(training, workers)
    assert (
        without["validation"]["exact_full_coverage"]
        and not without["efficacy_released"]
    )
    with pytest.raises(ValueError, match="six passed archive"):
        report.build_report(training, workers, require_complete=True)


def test_duplicate_worker_and_archive_binding_rejected(complete_fixture):
    training, workers, receipts = complete_fixture
    with pytest.raises(ValueError, match="Duplicate worker input"):
        report.build_report(training, [workers[0], workers[0]])
    broken = copy.deepcopy(receipts)
    broken["worker_2"]["files"]["rollouts.json"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="Local input differs"):
        report.build_report(training, workers, archive_receipts=broken)


def mutate_worker(tmp_path, source, change):
    import shutil

    root = tmp_path / "worker"
    shutil.copytree(source, root)
    change(root)
    # Resealing deliberately bypasses simple byte checks; semantic checks must
    # still catch corrupted noise, pairing, scope and physical row coverage.
    seal = json.loads((root / "completion_receipt.json").read_text())
    seal["files"] = {
        k: v for k, v in inventory(root).items() if k != "completion_receipt.json"
    }
    write(root / "completion_receipt.json", seal)
    return root


@pytest.mark.parametrize(
    "change",
    [
        "noise",
        "reset",
        "initial_success",
        "provider",
        "head_condition",
        "probe",
        "duplicate_row",
    ],
)
def test_semantic_mutations_fail_even_after_resealing(
    complete_fixture, tmp_path, change
):
    training, workers, _ = complete_fixture

    def alter(root):
        rows = json.loads((root / "rollouts.json").read_text())
        if change == "duplicate_row":
            rows[1] = rows[0]
            write(root / "rollouts.json", rows)
            return
        index = 3 if change == "head_condition" else 0
        row = rows[index]
        directory = root / row["relative_directory"]
        summary = json.loads((directory / "summary.json").read_text())
        events = [
            json.loads(line)
            for line in (directory / "events.jsonl").read_text().splitlines()
        ]
        if change == "noise":
            summary["generations"][0]["noise_sha256"] = H
            next(e for e in events if e["kind"] == "recipe_generation")["noise"][
                "sha256"
            ] = H
        elif change == "reset":
            summary["reset_audit"]["post_stabilization_state_sha256"] = "c" * 64
            summary["reset_audit"]["sha256"] = digest(
                {k: v for k, v in summary["reset_audit"].items() if k != "sha256"}
            )
        elif change == "initial_success":
            summary["initial_success"] = True
        elif change == "provider":
            summary["provider_calls"] = 1
        elif change == "head_condition":
            next(e for e in events if e["kind"] == "recipe_generation")["provenance"][
                "native_prefix_unchanged"
            ] = False
        elif change == "probe":
            summary["checks"]["feature_capture_parity"]["max_abs"] = 1e-8
            next(e for e in events if e["kind"] == "recipe_native_probes")["checks"] = (
                summary["checks"]
            )
        events[-1]["result"] = summary
        write(directory / "summary.json", summary)
        (directory / "events.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events)
        )
        rows[index] = {
            **summary,
            "relative_directory": row["relative_directory"],
            "summary_sha256": file_sha256(directory / "summary.json"),
        }
        write(root / "rollouts.json", rows)

    bad = mutate_worker(tmp_path, workers[0], alter)
    with pytest.raises(ValueError):
        report.build_report(training, [bad])


def test_training_cannot_admit_failed_anchor_or_evaluation_window(
    complete_fixture, tmp_path
):
    import shutil

    training, _, _ = complete_fixture
    root = tmp_path / "training"
    shutil.copytree(training, root)
    admission = json.loads((root / "admission.json").read_text())
    admission["fresh_anchor_rollouts"][0]["admitted"] = True
    write(root / "admission.json", admission)
    receipt = json.loads((root / "training_receipt.json").read_text())
    receipt["admission_sha256"] = file_sha256(root / "admission.json")
    write(root / "training_receipt.json", receipt)
    with pytest.raises(ValueError, match="Failed fresh anchors"):
        report.build_report(root, [])


def test_successful_fresh_anchor_admits_only_executed_windows(tmp_path):
    protocol = json.loads(
        (
            Path(report.__file__).parent / "configs/learned_correction_recipe_v1.json"
        ).read_text()
    )
    root = tmp_path / "training"
    make_training(root, protocol, successful_fresh=True)
    result = report.build_report(root, [])
    assert result["training"]["training_windows"] == 415
    assert (
        result["cost"]["training"]["frozen_feature_extraction"]["executed_rows"]
        == 1248 * 8
    )
    # A terminal two-action native anchor window still supplies five freshly
    # generated native velocity labels; it does not copy an unexecuted action.
    assert (
        result["cost"]["training"]["frozen_feature_extraction"]["anchor_rows"]
        == 160 * 40
    )
    assert (
        result["cost"]["training"]["native_anchor_collection"]["actions_executed"] == 22
    )


def test_held_selector_decision_lasts_25_actions_and_terminal_window_is_partial(
    tmp_path,
):
    entry = make_manifest("libero_goal_ood", [0], [1])["episodes"][0]
    summary = physical(tmp_path, "case", entry, "gated_flow_head", True, actions=31)
    parsed = report._rollout(
        report.Inputs(tmp_path, "synthetic"),
        "case",
        summary,
        entry,
        probe=False,
        schedules={},
        head_sha=HEAD,
    )
    assert parsed["decision_count"] == 2
    assert parsed["gate_active_actions"] == parsed["head_active_actions"] == 31
    assert parsed["velocity_evaluations"] == 70
    assert parsed["prefix_evaluations"] == 9
    events_path = tmp_path / "case/events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    generation = next(
        e
        for e in events
        if e["kind"] == "recipe_generation" and e["observation_step"] == 20
    )
    generation["choice"] = NATIVE
    summary["generations"][4]["choice"] = NATIVE
    events[-1]["result"] = summary
    events_path.write_text("".join(json.dumps(e) + "\n" for e in events))
    write(tmp_path / "case/summary.json", summary)
    with pytest.raises(ValueError, match="Held decision lifetime"):
        report._rollout(
            report.Inputs(tmp_path, "synthetic"),
            "case",
            summary,
            entry,
            probe=False,
            schedules={},
            head_sha=HEAD,
        )


def test_publication_has_portable_hash_bound_plots_and_no_weights(
    complete_fixture, tmp_path
):
    training, workers, receipts = complete_fixture
    value = report.build_report(
        training, workers, archive_receipts=receipts, require_complete=True
    )
    output = tmp_path / "published"
    manifest = report.write_report(value, output, include_videos=False)
    assert (output / "figures/outcomes.pdf").is_file()
    assert (output / "figures/training.png").is_file()
    for name, entry in manifest["files"].items():
        assert file_sha256(output / name) == entry["sha256"]
    assert not list(output.rglob("*.pt"))
    assert "same reset" in (output / "index.html").read_text()
    assert str(training) not in (output / "report.json").read_text()
    with pytest.raises(FileExistsError):
        report.write_report(value, output, include_videos=False)
