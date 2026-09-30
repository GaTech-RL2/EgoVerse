"""Synthetic contract checks, not measured robot or Astra performance."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal.demo_skill_agent import (
    SCHEMA_VERSION,
    parse_proposal,
    response_schema,
)
from astra_reversal.demo_skill_conditioning import InputSkillConditioner
from astra_reversal.demo_skill_experiment import load_protocol
from astra_reversal.demo_skill_program import (
    BehaviorLibrary,
    ProgramExecutor,
    composed_reference,
    has_effect,
    substituted_observation,
    validate_program,
)
from astra_reversal.records import digest


class Bank:
    bank_id = "a" * 64
    sources = {
        "10": {
            "source_id": "10",
            "frame_count": 30,
            "prompt": "put the bowl on the plate",
        },
        "18": {
            "source_id": "18",
            "frame_count": 30,
            "prompt": "put the bowl on the cabinet",
        },
    }

    def catalog(self):
        return list(self.sources.values())

    def arrays(self, source_id):
        return {"actions": np.arange(30 * 7, dtype=np.float32).reshape(30, 7) / 210}

    def observation(self, source_id, frame):
        return {
            "observation/image": np.full((224, 224, 3), frame, np.uint8),
            "observation/wrist_image": np.full((224, 224, 3), frame + 20, np.uint8),
            "observation/state": np.full(8, 0.01 * frame, np.float32),
        }


def stage(**changes):
    return {
        "source_id": "10",
        "start_frame": 0,
        "end_frame": 20,
        "alpha": 1.0,
        "playback": "advance",
        "state_mode": "live",
        "min_actions": 0,
        "max_actions": 25,
        "advance_when": "timeout",
        "threshold": 0.02,
        "language": None,
        "vision_operator": "none",
        "occlusion_box": None,
        **changes,
    }


def program(*stages):
    return {"native": False, "stages": list(stages)}


def live():
    return Bank().observation("10", 0)


def test_ordered_action_composition_and_terminal_repetition():
    bank = Bank()
    choice = {**stage(start_frame=17, end_frame=20), "frame": 17}
    result, receipt = composed_reference(bank, choice, np.zeros((10, 7), np.float32))
    np.testing.assert_array_equal(result[:3], bank.arrays("10")["actions"][17:20])
    np.testing.assert_array_equal(result[3:], np.repeat(result[2:3], 7, axis=0))
    assert receipt["terminal_frame_repetitions"] == 7


def test_substitution_preserves_live_feedback_and_can_replace_state():
    raw = live()
    before = digest(raw)
    choice = {**stage(vision_operator="pixels"), "frame": 12}
    altered = substituted_observation(Bank(), choice, raw)
    assert digest(raw) == before
    assert np.all(altered["observation/image"] == 12)
    assert np.all(altered["observation/wrist_image"] == 32)
    np.testing.assert_array_equal(
        altered["observation/state"], raw["observation/state"]
    )
    altered = substituted_observation(Bank(), {**choice, "state_mode": "donor"}, raw)
    np.testing.assert_allclose(altered["observation/state"], 0.12)
    neutral = substituted_observation(
        Bank(), {**choice, "state_mode": "donor", "alpha": 0}, raw
    )
    assert digest(neutral) == before


def test_phase_guards_use_actual_live_proprioception():
    executor = ProgramExecutor(
        program(stage(advance_when="eef_lift"), stage(source_id="18", max_actions=5)),
        Bank(),
        "action_composition",
    )
    current = live()
    assert executor.select(current, 0)["source_id"] == "10"
    assert executor.select(current, 5)["source_id"] == "10"
    current["observation/state"][2] = 0.03
    assert executor.select(current, 10)["source_id"] == "18"
    assert executor.select(current, 15) is None
    with pytest.raises(ValueError, match="consecutive"):
        executor.select(current, 25)


@pytest.mark.parametrize(
    "changes",
    [
        {"end_frame": 31},
        {"alpha": float("nan")},
        {"alpha": True},
        {"source_id": "missing"},
        {"min_actions": 3},
        {"state_mode": "donor"},
        {"vision_operator": "pixels"},
        {
            "language": {
                "operator": "tei",
                "source_a_id": "10",
                "source_b_id": "18",
                "alpha": 0.4,
            }
        },
    ],
)
def test_action_arm_rejects_invalid_or_cross_arm_programs(changes):
    with pytest.raises(ValueError):
        validate_program(
            program(stage(**changes)), Bank().catalog(), "action_composition"
        )


def test_tei_zero_is_active_tli_half_is_neutral():
    language = {
        "operator": "tei",
        "source_a_id": "10",
        "source_b_id": "18",
        "alpha": 0.0,
    }
    choice = stage(alpha=0, language=language)
    assert has_effect(choice, "input_skill_library")
    choice["language"] = {**language, "operator": "tli", "alpha": 0.5}
    assert not has_effect(choice, "input_skill_library")
    choice["language"]["alpha"] = 0
    assert has_effect(choice, "input_skill_library")
    choice["language"]["source_b_id"] = "10"
    assert not has_effect(choice, "input_skill_library")


class Policy:
    def __init__(self):
        self.calls = []

    def prepare(self, observation, observation_id, prompt):
        self.calls.append(("native", copy.deepcopy(observation), prompt))
        return "native-condition"

    def prepare_interpolated(self, observation, observation_id, prompt, **kwargs):
        self.calls.append(("interpolated", copy.deepcopy(observation), prompt, kwargs))
        return "edited-condition", kwargs


def test_input_skill_compiles_tei_and_occlusion_without_altering_live_data():
    policy = Policy()
    engine = InputSkillConditioner(policy, Bank(), {})
    choice = {
        **stage(
            vision_operator="occlusion",
            occlusion_box=[0, 0, 0.5, 0.5],
            language={
                "operator": "tei",
                "source_a_id": "10",
                "source_b_id": "18",
                "alpha": 0.25,
            },
        ),
        "frame": 0,
    }
    raw = live()
    original = digest(raw)
    condition, receipt, effective = engine.prepare(raw, "OOD task", choice)
    assert condition == "edited-condition" and digest(raw) == original
    assert np.all(effective["observation/image"][:112, :112] == 127)
    assert np.all(effective["observation/image"][112:] == 0)
    assert policy.calls[-1][2] == "OOD task"
    assert policy.calls[-1][3]["alpha"] == 0.25
    assert receipt["setting"]["vision_operator"] == "occlusion"


def test_neutral_tli_compiles_exact_native_and_missing_banks_reject():
    engine = InputSkillConditioner(Policy(), Bank(), {})
    text = {"operator": "tli", "source_a_id": "10", "source_b_id": "18", "alpha": 0.5}
    choice = {**stage(alpha=0, language=text), "frame": 0}
    assert engine.prepare(live(), "OOD task", choice)[0] == "native-condition"
    choice["language"] = {**text, "alpha": 0.2}
    with pytest.raises(ValueError, match="verified text bank"):
        engine.prepare(live(), "OOD task", choice)


def evidence(reset, split, success=True):
    return {
        "reset_id": reset,
        "split": split,
        "success": success,
        "trace_sha256": "b" * 64,
        "attempt_id": reset,
    }


def test_library_admission_requires_separate_validation_and_freezes(tmp_path):
    library = BehaviorLibrary(tmp_path, "a" * 64, "input_skill_library")
    p = program(stage(vision_operator="pixels"))
    first = library.record(
        task="put bowl on plate",
        program=p,
        hypothesis="test",
        evidence=[evidence("d0", "development")],
    )
    assert first["status"] == "observed_development_success"
    with pytest.raises(ValueError, match="separate resets"):
        library.record(
            task="test",
            program=p,
            hypothesis="test",
            evidence=[evidence("x", "development"), evidence("x", "validation")],
        )
    row = library.record(
        task="put bowl on plate",
        program=p,
        hypothesis="test",
        evidence=[
            evidence("d0", "development"),
            evidence("v0", "validation"),
            evidence("v1", "validation"),
        ],
    )
    assert row["status"] == "validated_on_new_resets"
    assert row["unseen_task_transfer"] == "not_established"
    assert row["skill_type"] == "pi05_input_settings"
    with pytest.raises(ValueError, match="evaluation"):
        library.record(
            task="test",
            program=p,
            hypothesis="test",
            evidence=[evidence("e0", "evaluation")],
        )
    assert library.freeze() == digest(library.data)
    with pytest.raises(ValueError, match="Frozen"):
        library.record(
            task="test",
            program=p,
            hypothesis="test",
            evidence=[evidence("new", "development")],
        )


def request():
    value = {
        "schema_version": SCHEMA_VERSION,
        "role": "program",
        "arm": "input_skill_library",
        "source_catalog": Bank().catalog(),
        "observation_step": 0,
        "text_bank_sources": ["10", "18"],
        "episode_id": "case",
        "attempt_id": "attempt",
        "request_index": 1,
        "request_id": "request",
    }
    value["request_fingerprint"] = digest(value)
    return value


def test_agent_response_is_bound_to_sources_and_request():
    req = request()
    response = {
        "selected_sources": ["10"],
        "program": program(stage(vision_operator="pixels")),
        "failure_hypothesis": "coarse hypothesis",
        "expected_effect": "test a different behavior",
    }
    parsed = parse_proposal(response, req)
    assert parsed["request_fingerprint"] == req["request_fingerprint"]
    assert parsed["decision_id"] == "request"
    response["program"]["stages"][0]["end_frame"] = 31
    with pytest.raises(ValueError):
        parse_proposal(response, req)
    req["source_catalog"][0]["prompt"] = "changed observation"
    with pytest.raises(ValueError, match="identity"):
        response_schema(req)


def test_new_relay_and_executor_family_dispatch():
    from astra_reversal import demo_skill_agent
    from astra_reversal.codex_executor import _module as executor_module
    from astra_reversal.codex_relay import _module as relay_module

    assert executor_module({"schema_version": SCHEMA_VERSION}) is demo_skill_agent
    assert relay_module("demo_skills") is demo_skill_agent


def test_full_protocol_preserves_all_tasks_and_disjoint_splits(tmp_path):
    config = load_protocol()
    assert len(config["suites"]) * len(config["task_ids"]) == 20
    assert len(config["source_task_ids"]) == 40
    config["evaluation_states"][0] = 0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="overlap"):
        load_protocol(path)


def test_full_action_path_uses_current_conditioning_and_neutral_identity(
    tmp_path, monkeypatch, spec
):
    from astra_reversal import demo_skill_experiment as experiment_module

    from .conftest import SyntheticPolicy

    policy = SyntheticPolicy()
    observed = []
    raw = live()
    original = digest(raw)
    entry = {
        "task_id": 1,
        "suite": "libero_goal_ood",
        "initial_state_id": 0,
        "seed": 97,
        "instruction": "synthetic OOD task",
        "episode_id": "synthetic:state0",
    }

    def run(_env, entry, _benchmark, callback, **_kwargs):
        observed.append(callback(raw, 0).copy())
        return {
            "episode_id": entry["episode_id"],
            "success": False,
            "reset_audit": {"synthetic": True},
            "actions_executed": 5,
            "snapshots": [{"step": 0, "observation": raw}],
        }

    monkeypatch.setattr(
        experiment_module.ActionSpec, "from_environment", lambda *_: spec
    )
    monkeypatch.setattr(experiment_module, "run_rollout", run)
    experiment = experiment_module.DemoSkillExperiment(
        policy,
        lambda *_: (None, SimpleNamespace(language=entry["instruction"]), None),
        None,
        Bank(),
        load_protocol(),
        tmp_path / "experiment",
        "action_composition",
        SimpleNamespace(records=[]),
    )
    experiment.run_program(entry, experiment_module.NATIVE, "native", "development")
    experiment.run_program(entry, program(stage(alpha=0)), "neutral", "development")
    np.testing.assert_array_equal(observed[0], observed[1])
    assert not policy.inverse_inputs
    experiment.run_program(entry, program(stage()), "edited", "development")
    # Constant synthetic flow is invertible analytically; it must recover recorded deltas.
    np.testing.assert_allclose(
        observed[2], Bank().arrays("10")["actions"][:10], atol=1e-6
    )
    assert len(policy.inverse_inputs) == 1
    np.testing.assert_array_equal(policy.inverse_inputs[0][1][..., 7:], 0)
    assert digest(raw) == original
    assert all(
        c.prompt == entry["instruction"]
        and digest({k: c.raw[k] for k in raw}) == original
        for c in policy.conditions
    )


def test_evaluation_only_completes_after_both_suites_and_all_pairs(
    tmp_path, monkeypatch
):
    from astra_reversal.demo_skill_experiment import NATIVE, DemoSkillExperiment

    experiment = DemoSkillExperiment(
        None,
        None,
        None,
        Bank(),
        load_protocol(),
        tmp_path / "study",
        "input_skill_library",
        SimpleNamespace(records=[]),
    )
    entries = []
    for suite in experiment.protocol["suites"]:
        experiment.selected[f"{suite}_task0"] = copy.deepcopy(NATIVE)
        entries.extend(
            {"suite": suite, "task_id": 0, "initial_state_id": reset}
            for reset in experiment.protocol["evaluation_states"]
        )

    def run(entry, _program, attempt_id, split):
        experiment.report["physical_rollouts"].append(
            {**entry, "attempt_id": attempt_id, "split": split}
        )

    monkeypatch.setattr(experiment, "run_program", run)
    experiment.freeze()
    experiment.evaluate(entries[:10])
    assert experiment.report["status"] == "evaluation"
    with pytest.raises(ValueError, match="missing or extra"):
        experiment.complete()
    experiment.evaluate(entries[10:])
    experiment.complete()
    assert experiment.report["status"] == "complete"


def test_history_shows_failure_when_another_development_reset_succeeded():
    from astra_reversal.demo_skill_experiment import NATIVE, archive_history

    outcomes = [
        {
            "attempt_id": f"attempt{i}",
            "episode_id": f"reset{i}",
            "success": success,
            "actions_executed": 25,
            "snapshots": [{"step": 25, "observation": live()}],
        }
        for i, success in enumerate((False, True))
    ]
    history = archive_history({"score": 1, "program": NATIVE, "results": outcomes})
    assert history["attempt_id"] == "attempt0" and not history["success"]
    assert history["development_successes"] == 1
    assert history["development_trials"][1]["success"]


def test_metrics_preserve_censoring_cost_scope_and_paired_counts(tmp_path):
    from astra_reversal.demo_skill_metrics import write_metrics

    report = {
        "status": "complete",
        "arm": "input_skill_library",
        "provider_records": [],
        "protocol": {
            "evaluation_states": [4],
            "evaluation_scope": "synthetic test only",
        },
        "tasks": {},
        "physical_rollouts": [],
    }
    for task_id in (0, 1):
        report["tasks"][f"libero_goal_ood_task{task_id}"] = {
            "provider_record_start": 0,
            "provider_record_end": 0,
            "first_success_provider_record_end": 0 if task_id == 0 else None,
            "first_success_candidate": 0 if task_id == 0 else None,
            "first_success_censored": task_id == 1,
            "validation_successes": 0,
            "candidates": [],
        }
        report["physical_rollouts"].extend(
            {
                "suite": "libero_goal_ood",
                "task_id": task_id,
                "split": "evaluation",
                "attempt_id": f"task{task_id}_eval4_{method}",
                "success": method == "composed",
                "wall_seconds": 2.0,
                "policy_seconds": 1.0,
            }
            for method in ("native", "composed")
        )
    result = write_metrics(report, tmp_path)
    assert result["native_successes"] == 0 and result["selected_successes"] == 2
    assert (
        result["paired_trials"] == 2
        and result["sr_difference_percentage_points"] == 100
    )
    assert result["tasks"][1]["search_to_first_success_usage"] is None
    assert result["search_usage"]["monetary_cost"] is None
    assert "offline demonstration curation" in result["cost_scope"]
    assert (tmp_path / "task_metrics.csv").read_text().count("\n") == 3
