"""Synthetic recording integrity tests; no GPU, simulator or provider traffic."""

import csv
import io
import json
import urllib.error
from dataclasses import asdict

import numpy as np
import pytest

from astra_reversal import intervention_report
from astra_reversal.astra_client import ClientError
from astra_reversal.config import BenchmarkConfig
from astra_reversal.intervention_agent import (
    ARM_CHANNELS,
    InterventionClient,
    summarize_calls,
)
from astra_reversal.intervention_report import build_report, write_outputs
from astra_reversal.intervention_search import (
    InterventionSearch,
    aggregate_reports,
    arm_summary,
    load_protocol,
)
from astra_reversal.interventions import ARMS
from astra_reversal.records import digest, file_sha256

from .conftest import SyntheticEnvironment, SyntheticPolicy

USAGE = {
    "prompt_tokens": 11,
    "completion_tokens": 3,
    "total_tokens": 14,
    "completion_tokens_details": {"reasoning_tokens": 2},
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text())


def rehash(value):
    value["sha256"] = digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def _legacy_success_recording(search, result):
    """Construct the frozen v1 force-attempt-2 recording from a synthetic run."""
    for arm in ARMS:
        value = result["arms"][arm]
        value["attempts"] = value["attempts"][:2]
        value["summary"] = arm_summary(value["attempts"], 5)
        value["summary"]["standalone_velocity_evaluations_through_success_or_cap"] = (
            value["summary"]["velocity_evaluations_through_success_or_cap"] + 1230
        )
        path = search.directory / f"{arm}_provider.jsonl"
        if path.exists():
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records = [row for row in records if row["iteration"] <= 2]
            for row in records:
                row["prompt_template_version"] = "astra-intervention-http-1"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            value["token_usage"] = summarize_calls(records)
    path = search.directory / "events.jsonl"
    retained = []
    for row in map(json.loads, path.read_text().splitlines()):
        iteration = row.get(
            "iteration",
            row.get("request", {}).get(
                "iteration", row.get("attempt", {}).get("iteration", 0)
            ),
        )
        if iteration <= 2:
            row["sequence"] = len(retained)
            retained.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in retained))
    physical = [result["baseline"], *result["controls"].values()] + [
        row for arm in ARMS for row in result["arms"][arm]["attempts"][1:]
    ]
    result["physical_cost"] = {
        "rollouts": sum(row["rollout_executed"] for row in physical),
        "simulated_actions": sum(row["actions_executed"] for row in physical),
        "velocity_evaluations": sum(row["velocity_evaluations"] for row in physical)
        + 1230,
        "token_usage": summarize_calls(
            [
                json.loads(line)
                for path in sorted(search.directory.glob("*_provider.jsonl"))
                for line in path.read_text().splitlines()
            ]
        ),
    }


def _manifest(suite, phase):
    benchmark = BenchmarkConfig.preset(suite)
    case_ids = (
        [[0, 1], [1, 1]] if phase == "development" else [[i, 0] for i in range(10)]
    )
    entries = []
    for task, state_index in case_ids:
        state = np.asarray([0.0, task, state_index], dtype=np.float64)
        model = {
            "body_pos": np.zeros((2, 3)),
            "body_quat": np.array([[1.0, 0, 0, 0]] * 2),
        }
        entries.append(
            {
                "episode_id": f"{suite}:seed19:task{task}:state{state_index}",
                "suite": suite,
                "task_id": task,
                "initial_state_id": state_index,
                "seed": 19,
                "instruction": "Place the test cup in the bowl.",
                "reset_state": state.tolist(),
                "reset_state_sha256": digest(state),
                "reset_model": {key: value.tolist() for key, value in model.items()},
                "reset_model_sha256": digest(model),
                "bddl_sha256": digest(f"bddl {task}"),
                "initially_successful": False,
                "prescribed_state_asset_sha256": digest("asset")
                if phase == "development"
                else None,
            }
        )
    return rehash(
        {
            "schema_version": "intervention_reset_v1",
            "evaluation_type": "online_adaptation_with_simulator_reset",
            "split": "development" if phase == "development" else "followup_adaptation",
            "seed": 19,
            "benchmark": asdict(benchmark),
            "libero_root": "/synthetic/libero",
            "reset_procedure": benchmark.reset_source
            + "_captured_before_stabilization",
            "cases": case_ids,
            "episodes": entries,
        }
    )


def _proposal(request):
    enabled = ARM_CHANNELS[request["arm"]]
    identity = {
        key: request[key]
        for key in (
            "schema_version",
            "episode_id",
            "iteration",
            "arm",
            "request_fingerprint",
        )
    }
    candidate = f"{request['arm']}_{request['iteration']}"
    return {
        **identity,
        "candidate_id": candidate,
        "best_candidate_id": candidate,
        "rationale": "Synthetic fixture proposal, not a benchmark measurement.",
        "language": {"target_text": "Move to the bowl.", "scale": 0.3}
        if "language" in enabled
        else None,
        "vision": [
            {
                "camera": "observation/image",
                "kind": "point",
                "coordinates": [2, 2],
                "gain": 0.3,
            }
        ]
        if "vision" in enabled
        else [],
        "noise": {
            "basis_id": request["basis_id"],
            "coefficients": [0.1] * 8,
            "perturbation_scale": 0.3,
        }
        if "noise" in enabled
        else None,
    }


class Opener:
    def __init__(self, scenario):
        self.scenario = scenario

    def open(self, request, timeout):
        context = json.loads(
            json.loads(request.data)["messages"][1]["content"][0]["text"]
        )["request"]
        arm, iteration = context["arm"], context["iteration"]
        if self.scenario == "all_http400" or (
            self.scenario == "accepted_no_rollout"
            and "language" not in ARM_CHANNELS[arm]
        ):
            raise urllib.error.HTTPError(
                request.full_url, 400, "Synthetic bad request", {}, io.BytesIO(b"{}")
            )
        if self.scenario == "mixed" and arm == "language_only":
            if iteration == 2:
                body = json.dumps(
                    {"model": "azure/openai/gpt-6-astra", "usage": USAGE}
                ).encode()
                raise urllib.error.HTTPError(
                    request.full_url, 503, "Synthetic unavailable", {}, io.BytesIO(body)
                )
            if iteration == 3:
                raise TimeoutError("Synthetic timeout")
        proposal = _proposal(context)
        if self.scenario == "neutral":
            if proposal["language"]:
                proposal["language"]["scale"] = 0.0
            if proposal["noise"]:
                proposal["noise"]["perturbation_scale"] = 0.0
            for mark in proposal["vision"]:
                mark["gain"] = 0.0
        envelope = {
            "model": "azure/openai/gpt-6-astra",
            "usage": USAGE,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(proposal),
                    },
                }
            ],
        }
        result = io.BytesIO(json.dumps(envelope).encode())
        result.status = 200
        return result


@pytest.fixture
def dataset(tmp_path, monkeypatch, spec):
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "synthetic-report-test-key")

    def make(phase="development", scenario="mixed", legacy=False):
        root = tmp_path / f"{phase}_{scenario}_{legacy}"
        protocol = load_protocol()
        if legacy:
            protocol.pop("development_embedding_probe")
            protocol.pop("development_stopping")
        policy = SyntheticPolicy()
        policy.metadata = {
            "frozen": True,
            "training_overlap": "unknown",
            "input_profile": "openpi_libero",
            "horizon": 10,
            "model_action_dim": 32,
            "synthetic_test_only": True,
        }
        policy.max_token_len = 200
        policy.prompt_length = lambda prompt, observation: (
            201 if scenario == "accepted_no_rollout" else 20
        )
        folders = []
        for worker in range(2 if phase == "development" else 8):
            suite = (
                "libero_10"
                if phase == "development"
                else ("libero_goal_ood", "libero_spatial_ood")[worker // 4]
            )
            target = {
                "suite": suite,
                "case_shard": worker if phase == "development" else worker % 4,
                "case_shards": 2 if phase == "development" else 4,
            }
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
                "packages": {"synthetic": "1.0"},
            }
            manifest = _manifest(suite, phase)
            entries = manifest["episodes"][
                target["case_shard"] :: target["case_shards"]
            ]
            write(directory / "runtime.json", runtime)
            write(directory / "protocol.json", protocol)
            write(directory / "checkpoint.json", policy.metadata)
            write(directory / "reset_manifest.json", manifest)
            write(
                directory / "frozen_plan.json",
                {
                    "runtime": runtime,
                    "protocol_sha256": digest(protocol),
                    "manifest_sha256": manifest["sha256"],
                    "assigned_episodes": [entry["episode_id"] for entry in entries],
                },
            )
            reports = []
            for entry in entries:
                search = InterventionSearch(
                    policy,
                    None,
                    BenchmarkConfig.preset(suite),
                    entry,
                    protocol,
                    directory / f"case_{entry['task_id']}_{entry['initial_state_id']}",
                    development=phase == "development",
                )
                search.spec = spec
                baseline_success = (
                    scenario not in ("mixed", "accepted_no_rollout")
                    or entry["task_id"] != 0
                )

                class Client(InterventionClient):
                    def __init__(self, **kwargs):
                        super().__init__(**kwargs)
                        self.opener = Opener(scenario)

                    def propose(self, request):
                        if scenario == "missing_key":
                            raise ClientError("NVIDIA_INFERENCE_API_KEY is not set")
                        return super().propose(request)

                search.client_type = Client

                def rollout(mode, iteration, proposal=None):
                    if "initialization" not in search.report:
                        noise = np.zeros((1, 10, 32), dtype=np.float32)
                        init = {
                            "passed": True,
                            "condition_id": digest("synthetic condition"),
                            "known_noise_sha256": digest(noise),
                            "recovered_noise_sha256": digest(noise),
                            "velocity_evaluations": 1230,
                            "errors": {
                                name: {"max_abs": 0.0, "rmse": 0.0}
                                for name in (
                                    "noise",
                                    "actions",
                                    "native_parity",
                                    "zero_embedding_hook_parity",
                                )
                            },
                        }
                        if not legacy:
                            probe = None
                            if phase == "development":
                                probe = {
                                    "kind": "fixed_numerical_probe_not_an_astra_proposal_or_rollout",
                                    "velocity_evaluations": 10,
                                    "conditioning": {
                                        "has_effect": True,
                                        "delta_frobenius": 0.1,
                                        "bound_frobenius": 0.25,
                                        "relative_rms": 0.1,
                                    },
                                    "output_difference": {"max_abs": 0.1, "rmse": 0.05},
                                }
                                search.recorder.event(
                                    "development_embedding_probe",
                                    probe=probe,
                                    actions=noise,
                                )
                                init["velocity_evaluations"] += 10
                            init["development_embedding_probe"] = probe
                        search.report["initialization"] = init
                        search.recorder.event(
                            "inversion_initialization",
                            known_noise=noise,
                            recovered_noise=noise,
                            action_spec=spec.as_dict(),
                            **init,
                        )
                    success = (
                        baseline_success
                        if mode == "reversal_identity"
                        else (
                            mode == "policy_fresh"
                            or (mode == "known_noise" and baseline_success)
                            or iteration
                            >= {
                                "random_noise": 3,
                                "noise_only": 2,
                                "language_only": 6,
                                "vision_only": 3,
                                "noise_language": 4,
                                "noise_vision": 3,
                                "language_vision": 4,
                                "joint": 3,
                            }.get(mode, 6)
                        )
                    )
                    reset = rehash(
                        {
                            **{
                                key: entry[key]
                                for key in (
                                    "episode_id",
                                    "seed",
                                    "reset_state_sha256",
                                    "reset_model_sha256",
                                    "bddl_sha256",
                                )
                            },
                            "post_stabilization_state_sha256": digest(
                                "stabilized state"
                            ),
                            "post_stabilization_model_sha256": entry[
                                "reset_model_sha256"
                            ],
                            "post_stabilization_observation_sha256": digest(
                                "stabilized images"
                            ),
                            "stabilization_steps": 10,
                            "initial_success": False,
                            "initial_terminated": False,
                        }
                    )
                    actions = 5 if success else 10
                    row = {
                        "episode_id": entry["episode_id"],
                        "success": success,
                        "actions_executed": actions,
                        "policy_replans": actions // 5,
                        "wall_seconds": 0.0,
                        "reset_seconds": 0.0,
                        "policy_seconds": 0.0,
                        "environment_seconds": 0.0,
                        "initial_success": False,
                        "captured_initial_success": False,
                        "zero_action_success": False,
                        "terminated": not success,
                        "action_budget": BenchmarkConfig.preset(
                            suite
                        ).task_action_budget,
                        "execute_steps": 5,
                        "reset_audit": reset,
                        "video_path": None,
                        "velocity_evaluations": actions * 2,
                        "clipped_values": 0,
                        "condition_preparation_seconds": 0.0,
                        "iteration": iteration,
                        "candidate_id": proposal["candidate_id"] if proposal else mode,
                        "proposal": proposal,
                        "rollout_executed": True,
                        "status": "success" if success else "terminated",
                    }
                    snapshots = [
                        {
                            "label": "first",
                            "step": 0,
                            "observation": SyntheticEnvironment().observe(),
                        }
                    ]
                    for step in range(0, actions, 5):
                        search.recorder.event(
                            "candidate_generation",
                            mode=mode,
                            iteration=iteration,
                            candidate_id=row["candidate_id"],
                            observation_step=step,
                            velocity_evaluations=10,
                        )
                    search.recorder.event("attempt", attempt=row, snapshots=snapshots)
                    return row, snapshots

                monkeypatch.setattr(search, "rollout", rollout)
                result = search.run()
                if legacy:
                    if baseline_success and phase == "development":
                        _legacy_success_recording(search, result)
                    for arm in ARMS:
                        if not (search.directory / f"{arm}_provider.jsonl").exists():
                            result["arms"][arm]["token_usage"] = None
                    write(search.directory / "summary.json", result)
                reports.append(result)
            write(directory / "aggregate.json", aggregate_reports(reports, protocol))
            if phase == "development" and not legacy:
                write(
                    directory / "development_validation.json",
                    {
                        row["episode_id"]: {
                            arm: any(
                                attempt["rollout_executed"]
                                for attempt in row["arms"][arm]["attempts"][1:]
                            )
                            for arm in ARMS
                        }
                        for row in reports
                    },
                )
            write(
                directory / "progress.json",
                {
                    "status": "complete",
                    "complete_cases": len(entries),
                    "assigned_cases": len(entries),
                    **target,
                },
            )
        return folders

    return make


def case_path(folders, worker=0):
    return next(folders[worker].glob("case_*/summary.json"))


def test_rescue_attempts_censoring_failed_calls_and_shared_physical_cost(dataset):
    report = build_report(dataset(), phase="development")
    summary = report["summary"]
    noise = summary["arms"]["noise_only"]
    random = summary["arms"]["random_noise"]
    language = summary["arms"]["language_only"]
    assert noise["successes_by_attempt"] == [1, 2, 2, 2, 2]
    assert noise["median_intervention_iterations_among_rescued_cases"] == 1
    assert random["median_intervention_iterations_among_rescued_cases"] == 2
    assert noise["median_attempt_to_success_among_successes"] == 1.5
    assert (
        noise["paired_rescue_vs_random_noise_by_attempt"][1]["counts"]["arm_only"] == 1
    )
    assert language["censored_cases"] == 1
    assert language["conditional_rescue"]["rescued_cases"] == 0
    assert language["median_intervention_iterations_among_rescued_cases"] is None
    assert language["token_usage_through_success_or_cap"]["provider_calls"] == 4
    assert language["token_usage_through_success_or_cap"]["tokens"]["input_tokens"] == {
        "sum": 33,
        "available_calls": 3,
        "missing_calls": 1,
        "complete": False,
    }
    assert (
        language["token_usage_through_success_or_cap"]["tokens"]["total_tokens"]["sum"]
        == 42
    )
    curve = noise["cumulative_token_usage_by_attempt_budget"]
    assert [row["through_success_or_cap"]["provider_calls"] for row in curve] == [
        0,
        1,
        1,
        1,
        1,
    ]
    assert curve[1]["actual_including_development_hooks"]["provider_calls"] == 2
    assert summary["physical_cost"]["initialization_velocity_evaluations"] == 2480
    standalone_total = sum(
        row["standalone_cost_through_success_or_cap"][
            "initialization_velocity_evaluations"
        ]
        for row in summary["arms"].values()
    )
    assert standalone_total == 8 * 2480
    assert report["audit"]["zero_action_successes"] == 0
    later_case = next(case for case in report["cases"] if case["task_id"] == 1)
    assert len(later_case["arms"]["language_only"]["attempts"]) == 4
    assert (
        later_case["arms"]["language_only"]["token_usage_actual"]["provider_calls"] == 3
    )
    assert (
        later_case["arms"]["language_only"]["tokens_to_first_success"]["provider_calls"]
        == 0
    )
    assert all(
        case["source"]["sha256"] == file_sha256(case["source"]["path"])
        for case in report["cases"]
    )


def test_completed_http400_records_are_failed_integration_not_intervention_success(
    dataset,
):
    report = build_report(
        dataset(scenario="all_http400", legacy=True), phase="development"
    )
    assert report["status"] == "complete_verified_recording"
    assert report["integration"]["status"] == "failed"
    assert report["integration"]["accepted_proposals"] == 0
    assert report["integration"]["executed_astra_intervention_rollouts"] == 0
    assert report["integration"]["baseline_failed_cases"] == 0
    physical = report["summary"]["physical_cost"]
    assert physical["rollouts"] == 8
    assert physical["token_usage"]["provider_calls"] == 14
    assert physical["token_usage"]["tokens"]["input_tokens"]["missing_calls"] == 14
    assert report["audit"]["legacy_null_usage_normalized_to_verified_zero"] == 2
    for arm in report["summary"]["arms"].values():
        assert arm["conditional_rescue"]["rescue_rate"] is None
        assert arm["median_intervention_iterations_among_rescued_cases"] is None


def test_explicit_pre_http_failure_is_counted_separately(dataset):
    report = build_report(
        dataset(scenario="missing_key", legacy=True), phase="development"
    )
    assert report["audit"]["pre_http_proposal_errors"] == 14
    assert report["summary"]["physical_cost"]["token_usage"]["provider_calls"] == 0
    assert report["integration"]["status"] == "failed"
    assert report["audit"]["legacy_null_usage_normalized_to_verified_zero"] == 16


def test_evaluation_requires_all_twenty_fixed_cases_and_never_forces_hooks(dataset):
    folders = dataset(phase="evaluation", scenario="baseline_success", legacy=True)
    report = build_report(folders, phase="evaluation")
    assert report["summary"]["cases"] == 20
    assert report["summary"]["physical_cost"]["rollouts"] == 60
    assert report["integration"]["status"] == "not_exercised"
    assert report["audit"]["legacy_null_usage_normalized_to_verified_zero"] == 160
    assert set(report["verified_full_reset_manifest_sha256"]) == {
        "libero_goal_ood",
        "libero_spatial_ood",
    }
    assert len({case["episode_id"] for case in report["cases"]}) == 20


@pytest.mark.parametrize(
    "change,match",
    [
        ("incomplete", "incomplete"),
        ("extra_case", "extra case"),
        ("baseline", "common baseline"),
        ("reset", "Full paired reset"),
        ("missing_events", "Missing or reordered event"),
        ("provider_duplicate", "Duplicate provider"),
        ("provider_missing", "token summary"),
        ("usage", "Normalized token ledger"),
        ("random", "matched seeded basis"),
        ("protocol", "protocol"),
    ],
)
def test_corrupt_or_incomplete_recordings_fail_closed(dataset, change, match):
    folders = dataset()
    path = case_path(folders)
    case = read(path)
    if change == "incomplete":
        status = read(folders[0] / "progress.json")
        status["status"] = "running"
        write(folders[0] / "progress.json", status)
    elif change == "extra_case":
        (folders[0] / "case_9_99").mkdir()
    elif change == "baseline":
        case["arms"]["joint"]["attempts"][0]["video_path"] = "different.mp4"
        write(path, case)
    elif change == "reset":
        audit = case["arms"]["noise_only"]["attempts"][1]["reset_audit"]
        audit["post_stabilization_observation_sha256"] = digest("different pixels")
        rehash(audit)
        write(path, case)
    elif change == "missing_events":
        events = path.with_name("events.jsonl")
        lines = events.read_text().splitlines()
        events.write_text("\n".join(lines[:3] + lines[4:]) + "\n")
    elif change in ("provider_duplicate", "provider_missing", "usage"):
        ledger = path.with_name("noise_only_provider.jsonl")
        if change == "provider_duplicate":
            ledger.write_text(ledger.read_text() * 2)
        elif change == "provider_missing":
            ledger.unlink()
        else:
            record = json.loads(ledger.read_text())
            record["token_usage"]["input_tokens"] += 1
            ledger.write_text(json.dumps(record) + "\n")
    elif change == "random":
        case["arms"]["random_noise"]["attempts"][1]["proposal"]["noise"][
            "perturbation_scale"
        ] = 0.123
        write(path, case)
    elif change == "protocol":
        protocol = read(folders[1] / "protocol.json")
        protocol["astra"]["model"] = "other-model"
        write(folders[1] / "protocol.json", protocol)
    with pytest.raises(ValueError, match=match):
        build_report(folders, phase="development")


@pytest.mark.parametrize("ulps", [1, 4, 8])
def test_random_direction_reconstruction_allows_only_declared_float64_roundoff(
    dataset, monkeypatch, ulps
):
    folders = dataset()
    original = intervention_report.random_noise_proposal

    def regenerated_with_other_blas_reduction(basis, rng):
        proposal = original(basis, rng)
        coefficients = np.asarray(proposal["coefficients"], dtype=np.float64)
        for _ in range(ulps):
            coefficients = np.nextafter(coefficients, np.inf)
        proposal["coefficients"] = coefficients.tolist()
        return proposal

    monkeypatch.setattr(
        intervention_report,
        "random_noise_proposal",
        regenerated_with_other_blas_reduction,
    )
    if ulps > intervention_report.RANDOM_COEFFICIENT_ULPS:
        with pytest.raises(ValueError, match="matched seeded basis"):
            build_report(folders, phase="development")
        return
    result = build_report(folders, phase="development")
    checks = [
        check
        for case in result["cases"]
        for check in case["random_noise_reconstruction"]
    ]
    assert checks
    assert all(check["basis_kind_and_scale_exact"] for check in checks)
    assert all(not check["coefficients_exact"] for check in checks)
    assert all(check["coefficient_max_ulp_difference"] == ulps for check in checks)
    assert all(check["coefficient_tolerance_ulps"] == 4 for check in checks)


@pytest.mark.parametrize(
    "coefficients",
    [
        ["0.0"] + [0.0] * 7,
        [False] + [0.0] * 7,
        [None] + [0.0] * 7,
        [float("nan")] + [0.0] * 7,
        [float("inf")] + [0.0] * 7,
        (0.0,) * 8,
    ],
)
def test_random_direction_reconstruction_requires_finite_numeric_list(coefficients):
    expected = {
        "kind": "low_rank",
        "basis_id": digest("basis"),
        "perturbation_scale": 0.15,
        "coefficients": [0.0] * 8,
    }
    row = {
        "iteration": 2,
        "rollout_executed": True,
        "proposal": {
            "language": None,
            "vision": [],
            "noise": {**expected, "coefficients": coefficients},
        },
    }
    with pytest.raises(ValueError, match="matched seeded basis"):
        intervention_report._random_noise_check(row, expected)


def test_full_unassigned_reset_geometry_must_match_between_workers(dataset):
    folders = dataset()
    path = folders[1] / "reset_manifest.json"
    manifest = read(path)
    unassigned = manifest["episodes"][0]
    unassigned["reset_model"]["body_pos"][0][0] = 0.1
    unassigned["reset_model_sha256"] = digest(
        {
            key: np.asarray(value, dtype=np.float64)
            for key, value in unassigned["reset_model"].items()
        }
    )
    rehash(manifest)
    write(path, manifest)
    plan = read(folders[1] / "frozen_plan.json")
    plan["manifest_sha256"] = manifest["sha256"]
    write(folders[1] / "frozen_plan.json", plan)
    with pytest.raises(ValueError, match="Full reset manifests differ"):
        build_report(folders, phase="development")


def test_rehashed_manifest_cannot_hide_inner_model_tamper(dataset):
    folders = dataset()
    path = folders[0] / "reset_manifest.json"
    manifest = read(path)
    manifest["episodes"][0]["reset_model"]["body_pos"][0][0] = 0.1
    write(path, rehash(manifest))
    with pytest.raises(ValueError, match="body poses hash"):
        build_report(folders, phase="development")


def test_explicit_phase_distinct_workers_and_new_output_directory(dataset, tmp_path):
    folders = dataset(scenario="all_http400", legacy=True)
    with pytest.raises(ValueError, match="exactly 8"):
        build_report(folders, phase="evaluation")
    with pytest.raises(ValueError, match="distinct"):
        build_report([folders[0], folders[0]], phase="development")
    report = build_report(folders, phase="development")
    output = tmp_path / "new_analysis"
    before = {
        path: file_sha256(path)
        for folder in folders
        for path in folder.rglob("*.json*")
    }
    write_outputs(output, report)
    assert {path.name for path in output.iterdir()} == {
        "report.json",
        "report.md",
        "arms.csv",
        "cases.csv",
        "budgets.csv",
        "attempts.csv",
    }
    assert "Astra integration: **failed**" in (output / "report.md").read_text()
    assert read(output / "report.json")["sha256"] == report["sha256"]
    with (output / "attempts.csv").open() as stream:
        attempts = list(csv.DictReader(stream))
    billable = [row for row in attempts if row["provider_calls"] == "1"]
    assert len(billable) == 14
    assert all(row["input_tokens_missing_calls"] == "1" for row in billable)
    assert all(file_sha256(path) == checksum for path, checksum in before.items())
    with pytest.raises(FileExistsError):
        write_outputs(output, report)


def test_missing_ledger_cannot_be_disguised_as_a_zero_cost_error(dataset):
    folders = dataset(scenario="missing_key", legacy=True)
    path = case_path(folders)
    report = read(path)
    report["arms"]["joint"]["attempts"][1]["error"] = "An unexplained transport error"
    write(path, report)
    events = path.with_name("events.jsonl")
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    for row in rows:
        if row["kind"] == "proposal_error" and row["arm"] == "joint":
            row["attempt"]["error"] = "An unexplained transport error"
    events.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="recognized pre-HTTP failure"):
        build_report(folders, phase="development")


def test_neutral_genuine_proposals_are_not_reported_as_nonzero_interventions(dataset):
    report = build_report(dataset(scenario="neutral"), phase="development")
    integration = report["integration"]
    assert integration["status"] == "passed"
    assert integration["accepted_proposals"] == 14
    assert integration["executed_astra_intervention_rollouts"] == 14
    assert (
        integration["executed_astra_revisions_with_nonzero_requested_parameters"] == 0
    )
    assert integration["fixed_development_embedding_probe_cases"] == 2


def test_provider_acceptance_without_any_executed_astra_candidate_is_not_integration_success(
    dataset,
):
    report = build_report(
        dataset(phase="evaluation", scenario="accepted_no_rollout"), phase="evaluation"
    )
    integration = report["integration"]
    assert integration["accepted_proposals"] == 32
    assert integration["executed_astra_intervention_rollouts"] == 0
    assert integration["status"] == "failed"
    assert report["summary"]["arms"]["language_only"]["censored_cases"] == 2
