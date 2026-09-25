"""Synthetic complete recordings; no GPU, simulator or provider requests."""

import io
import json
import urllib.error
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal import interpolation_agent, intervention_rollout
from astra_reversal.config import BenchmarkConfig
from astra_reversal.interpolation_agent import InterpolationClient
from astra_reversal.interpolation_catalog import (
    DATASET_REVISION,
    donor_catalog,
    donor_for,
    oracle_for,
)
from astra_reversal.interpolation_report import (
    _validate_inventory,
    build_report,
    write_outputs,
)
from astra_reversal.interpolation_search import (
    InterpolationSearch,
    aggregate_reports,
    load_protocol,
)
from astra_reversal.osmo.interpolation import assignment
from astra_reversal.records import digest, file_sha256

from .conftest import SyntheticPolicy

USAGE = {
    "prompt_tokens": 11,
    "completion_tokens": 3,
    "total_tokens": 14,
    "completion_tokens_details": {"reasoning_tokens": 2},
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def rehash(value):
    value["sha256"] = digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def inventory():
    donors = {}
    for index, item in enumerate(donor_catalog()):
        source_id = item["source_id"]
        donor = donor_for(source_id)
        bank = {
            "synthetic_test_only": True,
            "provenance": {
                "source_id": source_id,
                "source_prompt": donor.prompt,
                "dataset_revision": DATASET_REVISION,
                "frame_count": donor.all_frame_count,
                "episodes": [
                    {"episode_index": episode, "frame_count": frames}
                    for episode, frames in zip(
                        donor.episode_indices, donor.episode_frame_counts, strict=True
                    )
                ],
            },
        }
        bank["bank_id"] = digest(bank)
        donors[source_id] = {
            "bank": bank,
            "worker": index % 8,
            "files": [
                {
                    "name": name,
                    "key": f"synthetic/{source_id}/{name}",
                    "sha256": "c" * 64,
                    "bytes": 1,
                }
                for name in ("bank.npz", "manifest.json", "frames.jsonl")
            ],
        }
    return {
        "schema_version": "phase-bank-inventory-1.0",
        "workflow": "synthetic-test-only",
        "source_revision": "synthetic-test-only",
        "payload_sha256": "b" * 64,
        "donors": donors,
    }


def observation(step):
    return {
        "observation/image": np.full((8, 8, 3), step % 256, np.uint8),
        "observation/wrist_image": np.full((8, 8, 3), (step + 1) % 256, np.uint8),
        "observation/state": np.full(8, step / 1000, np.float32),
    }


def manifest(suite, protocol, phase):
    cases = (
        protocol["development_cases"][suite]
        if phase == "development"
        else protocol["evaluation_cases"]
    )
    benchmark = BenchmarkConfig.preset(suite)
    entries = []
    for task, index in cases:
        state = np.asarray([0, task, index], np.float64)
        model = {
            "body_pos": np.zeros((2, 3)),
            "body_quat": np.array([[1.0, 0, 0, 0]] * 2),
        }
        entries.append(
            {
                "episode_id": f"{suite}:seed{protocol['seed']}:task{task}:state{index}",
                "suite": suite,
                "task_id": task,
                "initial_state_id": index,
                "seed": protocol["seed"],
                "instruction": oracle_for(suite, task).task_name.replace("_", " "),
                "reset_state": state.tolist(),
                "reset_state_sha256": digest(state),
                "reset_model": {key: value.tolist() for key, value in model.items()},
                "reset_model_sha256": digest(model),
                "bddl_sha256": digest(f"synthetic {task}"),
                "initially_successful": False,
                "prescribed_state_asset_sha256": None,
            }
        )
    return rehash(
        {
            "schema_version": "intervention_reset_v1",
            "evaluation_type": "online_adaptation_with_simulator_reset",
            "split": "development" if phase == "development" else "followup_adaptation",
            "seed": protocol["seed"],
            "benchmark": asdict(benchmark),
            "libero_root": "/synthetic/libero",
            "reset_procedure": benchmark.reset_source
            + "_captured_before_stabilization",
            "cases": cases,
            "episodes": entries,
        }
    )


class Opener:
    def __init__(self, scenario):
        self.scenario = scenario

    def open(self, request, timeout):
        context = json.loads(
            json.loads(request.data)["messages"][1]["content"][0]["text"]
        )["request"]
        envelope = {"model": "azure/openai/gpt-6-astra", "usage": USAGE}
        if context["attempt_id"] == "astra_tli_2" and context["decision_index"] == 2:
            if self.scenario == "missing_usage":
                raise TimeoutError("synthetic timeout")
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "synthetic unavailable",
                {},
                io.BytesIO(json.dumps(envelope).encode()),
            )
        identity = (
            "schema_version",
            "episode_id",
            "attempt_id",
            "decision_index",
            "observation_step",
            "interpolation_mode",
            "request_fingerprint",
        )
        proposal = {
            **{key: context[key] for key in identity},
            "decision_id": f"{context['attempt_id']}_decision_{context['decision_index']}",
            "source_a_id": "13",
            "source_b_id": "17",
            "alpha": 0.1 * context["decision_index"],
            "observed_phase": "Synthetic fixture phase",
            "rationale": "Synthetic fixture, not a benchmark outcome.",
            "vision": [
                {
                    "camera": "observation/image",
                    "kind": "point",
                    "coordinates": [2, 3],
                    "gain": 0.4,
                }
            ]
            if context["vision_enabled"]
            else [],
        }
        envelope["choices"] = [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(proposal)},
            }
        ]
        result = io.BytesIO(json.dumps(envelope).encode())
        result.status = 200
        return result


@pytest.fixture
def dataset(tmp_path, monkeypatch, spec):
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "synthetic-phase-report-key")
    monkeypatch.setattr(
        "astra_reversal.interpolation_search.ActionSpec.from_environment",
        lambda *args: spec,
    )

    def make(phase="development", scenario="success"):
        protocol = load_protocol()
        if phase == "development":
            protocol["seed"] = 19
        root = tmp_path / f"{phase}-{scenario}"
        opener = Opener(scenario)

        class Client(InterpolationClient):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.opener = opener

            def propose(self, request):
                if (
                    scenario == "preflight"
                    and request["attempt_id"] == "astra_tli_vision_2"
                    and request["decision_index"] == 1
                ):
                    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY")
                    try:
                        return super().propose(request)
                    finally:
                        monkeypatch.setenv(
                            "NVIDIA_INFERENCE_API_KEY", "synthetic-phase-report-key"
                        )
                return super().propose(request)

        monkeypatch.setattr(interpolation_agent, "InterpolationClient", Client)
        folders = []
        for worker in range(3 if phase == "development" else 8):
            target = assignment(phase, worker)
            suite = target["suite"]
            benchmark = BenchmarkConfig.preset(suite)
            directory = root / f"worker_{worker}"
            directory.mkdir(parents=True)
            folders.append(directory)
            runtime = {
                "phase": phase,
                "worker": worker,
                "assignment": target,
                "tf32": False,
                "gpu": "synthetic L40S",
                "payload_sha256": "a" * 64,
                "workflow": "synthetic-test-only",
                "packages": {"fixture": "1"},
            }
            policy = SyntheticPolicy()
            policy.metadata = {
                "frozen": True,
                "training_overlap": "unknown",
                "input_profile": "openpi_libero",
                "horizon": 10,
                "model_action_dim": 32,
                "synthetic_test_only": True,
            }
            policy.observation_image_size = 8

            def prepare_interpolated(raw, observation_id, prompt, **kwargs):
                return policy.prepare(raw, observation_id, prompt), {
                    "has_effect": kwargs["operator"] == "tei" or kwargs["alpha"] != 0.5
                }

            policy.prepare_interpolated = prepare_interpolated
            reset_manifest = manifest(suite, protocol, phase)
            entries = (
                [
                    row
                    for row in reset_manifest["episodes"]
                    if row["task_id"] == target["task_id"]
                ]
                if phase == "development"
                else reset_manifest["episodes"][
                    target["case_shard"] :: target["case_shards"]
                ]
            )
            for filename, value in (
                ("runtime.json", runtime),
                ("protocol.json", protocol),
                ("checkpoint.json", policy.metadata),
                ("reset_manifest.json", reset_manifest),
                ("bank_inventory.json", inventory()),
            ):
                write(directory / filename, value)
            write(
                directory / "frozen_plan.json",
                {
                    "runtime": runtime,
                    "protocol_sha256": digest(protocol),
                    "manifest_sha256": reset_manifest["sha256"],
                    "bank_inventory_sha256": file_sha256(
                        directory / "bank_inventory.json"
                    ),
                    "assigned_episodes": [row["episode_id"] for row in entries],
                },
            )
            reports = []
            for entry in entries:
                search = InterpolationSearch(
                    policy,
                    lambda *args: (SimpleNamespace(), None, None),
                    benchmark,
                    entry,
                    protocol,
                    directory / f"case_{entry['task_id']}_{entry['initial_state_id']}",
                    banks={row["source_id"]: None for row in donor_catalog()},
                    catalog=donor_catalog(),
                    oracle=oracle_for(suite, entry["task_id"]).metadata(),
                    development=phase == "development",
                )

                def initialize_noise(raw, action_spec):
                    search.known = np.zeros((1, 10, 32), np.float32)
                    search.recovered = np.full((1, 10, 32), 0.01, np.float32)
                    search.report["initialization"] = {
                        "passed": True,
                        "condition_id": digest("native"),
                        "known_noise_sha256": digest(search.known),
                        "recovered_noise_sha256": digest(search.recovered),
                        "errors": {
                            key: {"rmse": 0.0, "max_abs": 0.0}
                            for key in (
                                "noise",
                                "actions",
                                "native_parity",
                                "zero_embedding_hook_parity",
                            )
                        },
                        "velocity_evaluations": 1230,
                        "development_embedding_probe": None,
                    }
                    search.recorder.event(
                        "inversion_initialization",
                        observation=raw,
                        known_noise=search.known,
                        recovered_noise=search.recovered,
                        action_spec=action_spec.as_dict(),
                        **search.report["initialization"],
                    )

                def check_interpolation(raw):
                    gate = {
                        "passed": True,
                        "status": "passed",
                        "complete": True,
                        "velocity_evaluations_complete": True,
                        "velocity_evaluations": 50,
                        "reference_velocity_evaluations": 10,
                        "solver": protocol["execution_solver"],
                        "known_noise_sha256": digest(search.known),
                        "native_condition_id": digest("native"),
                        "observation_id": digest(raw),
                        "probe_kind": "fixed_weighted_numerical_probe_without_environment_actions",
                        "checks": {},
                    }
                    search.recorder.event(
                        "interpolation_gate_reference",
                        observation=raw,
                        observation_id=gate["observation_id"],
                        condition_id=gate["native_condition_id"],
                        latent=search.known,
                        generated_actions=search.known,
                        decoded_actions=np.zeros((10, 7), np.float32),
                        solver=gate["solver"],
                        velocity_evaluations=10,
                    )
                    for label, operator, alpha, effect in (
                        ("tei_identical_sources", "tei", 0.37, False),
                        ("tli_zero_residual", "tli", 0.5, False),
                        ("tei_nonzero_oracle_sources", "tei", 0, True),
                        ("tli_nonzero_oracle_banks", "tli", 0, True),
                    ):
                        row = {
                            "passed": True,
                            "status": "passed",
                            "operator": operator,
                            "alpha": alpha,
                            "expected_effect": "nonzero" if effect else "identity",
                            "source_ids": [
                                search.oracle["source_a_id"],
                                search.oracle["source_b_id"],
                            ]
                            if effect
                            else None,
                            "condition_id": digest("edited")
                            if effect
                            else gate["native_condition_id"],
                            "provenance": {"has_effect": effect},
                            "velocity_evaluations": 10,
                            **{
                                key: {"rmse": float(effect), "max_abs": float(effect)}
                                for key in (
                                    "errors",
                                    "controlled_channel_errors",
                                    "decoded_action_errors",
                                )
                            },
                        }
                        gate["checks"][label] = row
                        search.recorder.event(
                            "interpolation_gate_check",
                            label=label,
                            observation_id=gate["observation_id"],
                            latent_sha256=gate["known_noise_sha256"],
                            generated_actions=search.known,
                            decoded_actions=np.zeros((10, 7), np.float32),
                            **row,
                        )
                    search.report["interpolation_gate"] = gate
                    search.recorder.event("interpolation_gate", **gate)
                    search.interpolation_checked = True

                monkeypatch.setattr(search, "initialize_noise", initialize_noise)
                monkeypatch.setattr(search, "check_interpolation", check_interpolation)

                def run_rollout(
                    env,
                    current_entry,
                    current_benchmark,
                    callback,
                    *,
                    execute_steps,
                    action_budget,
                    expected_reset,
                    policy_image_size,
                    video_path,
                ):
                    attempt_id = Path(video_path).stem
                    mode, iteration = attempt_id.rsplit("_", 1)
                    iteration = int(iteration)
                    mixed = (
                        scenario == "mixed"
                        and current_entry["suite"] == "libero_goal_ood"
                    )
                    success = (
                        not mixed
                        or mode in CONTROLS
                        or (mode == "random_noise" and iteration == 3)
                        or (mode == "astra_tei" and iteration == 3)
                        or mode == "astra_tli"
                    )
                    actions = 30 if mode.startswith("astra_") else 5
                    for step in range(0, actions, 5):
                        callback(observation(step), step)
                    audit = rehash(
                        {
                            **{
                                key: current_entry[key]
                                for key in (
                                    "episode_id",
                                    "seed",
                                    "reset_state_sha256",
                                    "reset_model_sha256",
                                    "bddl_sha256",
                                )
                            },
                            **{
                                f"post_stabilization_{key}_sha256": digest(key)
                                for key in ("state", "model", "observation")
                            },
                            "stabilization_steps": 10,
                            "initial_success": False,
                            "initial_terminated": False,
                        }
                    )
                    assert expected_reset in (None, audit)
                    return {
                        "episode_id": current_entry["episode_id"],
                        "success": success,
                        "initial_success": False,
                        "zero_action_success": False,
                        "captured_initial_success": False,
                        "terminated": not success,
                        "actions_executed": actions,
                        "policy_replans": actions // 5,
                        "wall_seconds": 1.0,
                        "reset_seconds": 0.01,
                        "policy_seconds": 0.1,
                        "environment_seconds": 0.2,
                        "reset_audit": audit,
                        "execute_steps": 5,
                        "action_budget": 300,
                        "snapshots": [
                            {
                                "label": "first",
                                "step": 0,
                                "observation": observation(0),
                            },
                            {
                                "label": "last",
                                "step": actions,
                                "observation": observation(actions),
                            },
                        ],
                    }

                monkeypatch.setattr(intervention_rollout, "run_rollout", run_rollout)
                reports.append(search.run())
            write(directory / "aggregate.json", aggregate_reports(reports, protocol))
            write(directory / "progress.json", {"status": "complete", **target})
        return folders

    return make


CONTROLS = ("known_noise", "policy_fresh")


def test_development_forced_extras_are_physical_cost_not_rescue(dataset, tmp_path):
    report = build_report(dataset(), phase="development")
    group = report["groups"]["pooled"]
    assert group["baseline_successes"] == group["cases"] == 3
    assert group["physical_cost"]["rollouts"] == 30
    assert group["physical_cost"]["token_usage"]["provider_calls"] == 18
    assert group["physical_cost"]["token_usage"]["tokens"]["total_tokens"]["sum"] == 252
    assert group["physical_cost"]["token_usage"]["failed_calls"] == 3
    assert group["execution"]["actions_with_held_text_after_failed_call"] == 15
    for arm in group["arms"].values():
        assert arm["rescues"] == 0
        assert arm["median_rollout_revisions_among_rescues"] is None
        assert arm["development_extra_rollouts_after_success"] == 3
        assert arm["success_by_budget"][-1]["provider"]["provider_calls"] == 0
    out = tmp_path / "rendered"
    write_outputs(out, report)
    assert len(list(out.glob("*.png"))) == 3
    assert "252" in (out / "report.md").read_text()
    with pytest.raises(FileExistsError):
        write_outputs(out, report)


def test_conditional_rescue_censoring_and_failed_call_tokens(dataset):
    report = build_report(dataset(scenario="mixed"), phase="development")
    group = report["groups"]["pooled"]
    assert group["baseline_failed_cases"] == 1
    assert group["arms"]["astra_tei"]["median_rollout_revisions_among_rescues"] == 2
    assert (
        group["arms"]["astra_tei"]["median_decisions_through_success_among_rescues"]
        == 4
    )
    assert (
        group["arms"]["astra_tei"][
            "median_total_tokens_among_rescues_with_complete_usage"
        ]
        == 56
    )
    assert (
        group["arms"]["astra_tli"][
            "median_total_tokens_among_rescues_with_complete_usage"
        ]
        == 28
    )
    assert group["arms"]["astra_tli_vision"]["censored_without_success"] == 1
    assert group["arms"]["oracle_tli"]["attempt_cap"] == 2
    assert len(group["arms"]["oracle_tli"]["success_by_budget"]) == 2
    assert (
        group["arms"]["oracle_tli"]["paired_random_at_same_cap"]["counts"]["both_fail"]
        == 1
    )


def test_preflight_slots_are_not_billable_and_unknown_usage_remains_partial(dataset):
    preflight = build_report(dataset(scenario="preflight"), phase="development")[
        "groups"
    ]["pooled"]
    usage = preflight["physical_cost"]["token_usage"]
    assert usage["client_attempts"] == 18
    assert usage["provider_calls"] == 15
    assert usage["preflight_failures"] == 3
    assert usage["tokens"]["total_tokens"]["sum"] == 210
    assert preflight["execution"]["native_condition_fallback_actions"] == 75
    unknown = build_report(dataset(scenario="missing_usage"), phase="development")[
        "groups"
    ]["pooled"]["physical_cost"]["token_usage"]
    assert unknown["provider_calls"] == 18
    assert unknown["tokens"]["total_tokens"]["missing_calls"] == 3
    assert unknown["tokens"]["total_tokens"]["complete"] is False


def test_evaluation_requires_all20_seed29_cases_and_counts_shared_baseline_once(
    dataset,
):
    folders = dataset(phase="evaluation")
    with pytest.raises(ValueError, match="Incomplete"):
        build_report(folders[:-1], phase="evaluation")
    report = build_report(folders, phase="evaluation")
    assert report["seed"] == 29 and len(report["cases"]) == 20
    group = report["groups"]["pooled"]
    assert group["physical_cost"]["rollouts"] == 60
    assert group["physical_cost"]["velocity_evaluations"] == 26200
    assert group["physical_cost"]["token_usage"]["provider_calls"] == 0
    assert group["arms"]["astra_tei"]["development_extra_rollouts_after_success"] == 0


def test_self_consistent_bank_identity_cannot_change_prescribed_demo_coverage():
    value = inventory()
    _validate_inventory(value)
    bank = value["donors"]["10"]["bank"]
    bank["provenance"]["episodes"][0]["frame_count"] -= 1
    bank["bank_id"] = digest(
        {key: item for key, item in bank.items() if key != "bank_id"}
    )
    with pytest.raises(ValueError, match="prescribed standard demonstrations"):
        _validate_inventory(value)


@pytest.mark.parametrize(
    "corruption",
    [
        "baseline",
        "provider",
        "manifest",
        "gate",
        "bank_inventory",
        "outcome",
        "feedback_pixels",
        "vision_expiry",
        "physical_cost",
    ],
)
def test_integrity_corruptions_fail_closed(dataset, corruption):
    folders = dataset()
    directory = folders[0]
    case = next(directory.glob("case_*"))
    path = case / "summary.json"
    value = json.loads(path.read_text())
    if corruption == "baseline":
        value["arms"]["astra_tei"]["attempts"][0]["reset_audit"][
            "post_stabilization_state_sha256"
        ] = "b" * 64
    elif corruption == "provider":
        provider = case / "astra_tei_2_provider.jsonl"
        provider.write_text("")
    elif corruption == "manifest":
        manifest_path = directory / "reset_manifest.json"
        changed = json.loads(manifest_path.read_text())
        changed["episodes"][0]["reset_model"]["body_pos"][0][0] = 1
        write(manifest_path, rehash(changed))
    elif corruption == "gate":
        value["interpolation_gate"]["checks"]["tli_nonzero_oracle_banks"][
            "decoded_action_errors"
        ]["max_abs"] = 0
    elif corruption == "bank_inventory":
        inventory_path = directory / "bank_inventory.json"
        inventory = json.loads(inventory_path.read_text())
        inventory["donors"].pop("10")
        write(inventory_path, inventory)
    elif corruption in ("feedback_pixels", "vision_expiry"):
        event_path = case / "events.jsonl"
        events = [json.loads(line) for line in event_path.read_text().splitlines()]
        changed = next(
            row
            for row in events
            if row["kind"] == "phase_generation"
            and row["mode"] == "astra_tli_vision"
            and row["observation_step"] == (0 if corruption == "feedback_pixels" else 5)
        )
        if corruption == "feedback_pixels":
            changed["observation"]["observation/image"]["sha256"] = "b" * 64
        else:
            changed["vision"] = [
                {
                    "camera": "observation/image",
                    "kind": "point",
                    "coordinates": [2, 3],
                    "gain": 0.4,
                }
            ]
        event_path.write_text("".join(json.dumps(row) + "\n" for row in events))
    elif corruption == "physical_cost":
        value["physical_cost"]["velocity_evaluations"] += value["baseline"][
            "velocity_evaluations"
        ]
    else:
        value["arms"]["astra_tei"]["summary"]["success"] = False
    write(path, value)
    with pytest.raises((ValueError, KeyError)):
        build_report(folders, phase="development")
