"""Replication keeps matched starts; speed rewards success, not early failure."""

import copy
import json
from pathlib import Path

import pytest
from test_analysis import fixture
from test_contract import Source, proxy

from astra_reversal.hardware_interface import provider
from astra_reversal.hardware_interface.protocol import load_manifest, schedule, validate
from astra_reversal.hardware_interface.provider import Meter, ModelFailure, Session
from astra_reversal.hardware_interface.replication import deepseek_design
from astra_reversal.hardware_interface.runner import run_trial
from astra_reversal.hardware_interface.speed import analyze_speed, timing
from experiments.libero_hardware_interface.tools.parallel_plan import (
    create_fresh_plan,
    validate_plan,
)

STUDY = Path(__file__).resolve().parents[3] / "experiments/libero_hardware_interface"


@pytest.mark.parametrize("suffix", ["", "-adaptive"])
def test_replication_preserves_paired_design_and_original_manifest(suffix):
    original = load_manifest(STUDY / ("preregistration" + suffix + ".yaml"))
    snapshot = copy.deepcopy(original)
    new = deepseek_design(original)
    saved = load_manifest(STUDY / ("preregistration-deepseek" + suffix + ".yaml"))
    assert saved == new and original == snapshot
    for key in (
        "conditions",
        "limits",
        "interface_evolution",
        "smoke",
        "libero_commit",
        "task_ids",
        "pilot_indices",
        "confirmatory_indices",
        "environment_seed",
        "randomization_seed",
        "replicates",
        "image_size",
        "settling_steps",
    ):
        assert new.get(key) == original.get(key)
    catalog = [
        {"task_id": t, "init_state_hashes": {str(i): f"{t}-{i}" for i in range(15)}}
        for t in range(10)
    ]
    before, after = schedule(original, catalog), schedule(new, catalog)
    for old, row in zip(before, after):
        assert {k: v for k, v in old.items() if not k.startswith("model_")} == {
            k: v for k, v in row.items() if not k.startswith("model_")
        }
    plan = create_fresh_plan(
        new, after, source="source", source_sha256="a" * 64, cohort="fresh"
    )
    validate_plan(plan, new, after)
    assert plan["parent_trial_ids"] == []
    assert sum(len(s["trial_ids"]) for s in plan["shards"]) == 50 * len(
        new["conditions"]
    )


def test_cannot_silently_substitute_older_model_or_astra_observer():
    manifest = load_manifest(STUDY / "preregistration-deepseek.yaml")
    for field, value in (
        ("identifier", "nvidia/deepseek-ai/deepseek-v4-flash"),
        ("base_url", "https://inference-api.nvidia.com/v1"),
        ("context_truncation", "auto"),
    ):
        bad = copy.deepcopy(manifest)
        bad["model"][field] = value
        if field == "identifier":
            bad["observer_model"] = value
        with pytest.raises(ValueError, match="unrecognized_model_replication"):
            validate(bad)
    manifest["observer_model"] = "openai/openai/gpt-6-astra"
    with pytest.raises(ValueError, match="actor_observer_model_mismatch"):
        validate(manifest)


def test_deepseek_preserves_reasoning_and_image_tool_roundtrip(tmp_path):
    manifest = load_manifest(STUDY / "preregistration-deepseek.yaml")
    p = proxy(tmp_path, limits=validate(manifest))
    requests = []
    reasoning = {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": "fixture reasoning"}],
    }

    def post(path, body, timeout):
        assert path == "responses"
        requests.append(copy.deepcopy(body))
        return {
            "model": "deepseek-flash",
            "status": "completed",
            "usage": {"input_tokens": 30, "output_tokens": 10},
            "output": [
                reasoning,
                {
                    "type": "function_call",
                    "name": "finish",
                    "arguments": "{}",
                    "call_id": str(len(requests)),
                },
            ],
        }

    session = Session(manifest["model"], p.limits, Meter(None), p.events, post=post)
    session.request("same prompt", [{"type": "function", "name": "finish"}])
    session.tool_result("1", p.observe(p.camera_keys))
    session.request("same prompt", [{"type": "function", "name": "finish"}])
    for body in requests:
        assert not {
            "truncation",
            "store",
            "include",
            "parallel_tool_calls",
            "max_output_tokens",
        } & set(body)
        assert body["instructions"].endswith(
            "Issue exactly one tool call per response."
        )
        assert body["tool_choice"] == "required"
    assert requests[1]["input"][0] == reasoning
    images = requests[1]["input"][-1]["content"]
    assert sum(item["type"] == "input_image" for item in images) == 2


def test_first_success_time_precedes_trial_end_and_excludes_setup(tmp_path):
    manifest = load_manifest(STUDY / "preregistration-deepseek.yaml")
    clock = [100.0]
    p = proxy(tmp_path, success_at=1, limits=validate(manifest), clock=lambda: clock[0])

    def post(path, body, timeout):
        clock[0] += 7
        return {
            "model": "deepseek-flash",
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 10},
            "output": [
                {
                    "type": "function_call",
                    "name": "step",
                    "call_id": "step",
                    "arguments": json.dumps(
                        {"action": [0.0] * 7, "repeat_steps": 1, "observation_step": 0}
                    ),
                }
            ],
        }

    clock[0] += 80  # setup/reset is outside the actor clock
    outcome = run_trial(
        p, Source(), "B0", "goal", manifest["model"], scratch=None, post=post
    )
    assert outcome["success"]
    assert outcome["first_success_wall_seconds"] == 7
    assert outcome["simulated_control_seconds"] == 0.05
    assert outcome["first_success_wall_seconds"] <= outcome["wall_s"]


def test_retry_backoff_is_separate_from_provider_roundtrip(tmp_path, monkeypatch):
    manifest = load_manifest(STUDY / "preregistration-deepseek.yaml")
    p = proxy(tmp_path, limits=validate(manifest))
    clock, attempts = [0.0], []
    monkeypatch.setattr(provider.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        provider.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    def post(path, body, timeout):
        attempts.append(body)
        clock[0] += 3
        if len(attempts) == 1:
            raise ModelFailure("provider_http_429")
        return {
            "model": "deepseek-flash",
            "status": "completed",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "output": [],
        }

    meter = Meter(None)
    Session(manifest["model"], p.limits, meter, p.events, post=post).request(
        "prompt", []
    )
    assert meter.usage["actor"]["provider_wall_seconds"] == 6
    assert meter.usage["actor"]["client_retry_backoff_seconds"] == 2
    assert meter.unknown == 1
    assert attempts[0] is attempts[1]


def test_fast_failure_never_becomes_fast_success_and_missing_stays_visible():
    manifest, rows = fixture()
    for row in rows:
        row["wall_s"] = 1 if row["condition"] == "B0" else 600
        if row["condition"] == "B0":
            row["success"] = False
            row["terminal_reason"] = "TASK_FAILURE"
    result = analyze_speed(rows, manifest, split="pilot")
    assert result["arms"]["B0"]["successful_wall_seconds"] == []
    assert all(
        p["fraction_of_assigned"] == 0 for p in result["arms"]["B0"]["success_by_time"]
    )
    assert result["paired"]["F_vs_B0"]["median_reference_over_other"] is None
    assert result["arms"]["F"]["time_to_fraction_of_assigned"]["0.5"] is None
    partial = analyze_speed(rows[:3], manifest, split="pilot")
    assert partial["arms"]["F"]["assigned"] == 50
    assert partial["arms"]["F"]["missing"] == 49
    assert (
        max(p["fraction_of_assigned"] for p in partial["arms"]["F"]["success_by_time"])
        == 0.02
    )


def test_speed_requires_verified_matched_unique_starts():
    manifest, rows = fixture()
    for mutation, message in (
        ("audit", "verified"),
        ("state", "unmatched"),
        ("duplicate", "duplicate"),
    ):
        bad = copy.deepcopy(rows)
        if mutation == "audit":
            bad[0]["independent_evaluation_passed"] = False
        elif mutation == "state":
            bad[0]["init_state_hash"] = "wrong"
        else:
            bad.append(bad[0])
        with pytest.raises(ValueError, match=message):
            analyze_speed(bad, manifest, split="pilot")


def test_timing_cannot_double_count_nested_tools_or_hide_unknowns():
    row = {
        "wall_s": 20,
        "provider_wall_seconds": 10,
        "client_admission_seconds": 2,
        "client_retry_backoff_seconds": 3,
        "simulator_and_step_evaluator_seconds": 4,
        "tool_latency_seconds": 100,
    }
    assert timing(row)["other_local_seconds"] == 1
    row["provider_wall_seconds"] = 40
    with pytest.raises(ValueError, match="overlapping"):
        timing(row)
    del row["client_admission_seconds"]
    assert timing(row)["other_local_seconds"] is None
