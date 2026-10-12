"""Pair integrity and source routing must fail closed before any GPU rollout."""

import copy
from types import SimpleNamespace

import pytest

from astra_reversal.hardware_interface.liberomax import (
    EVENTS,
    Environment,
    commissioning_cases,
    event_probe,
    source_of,
    verify_pair,
)
from astra_reversal.hardware_interface.protocol import (
    design,
    validate,
    without_resource_budgets,
)


def pair():
    control = {
        "arm": "control",
        "scenario_id": "case",
        "scenario_seed": 42,
        "episode_index": 0,
        "task_suite_name": "libero_10",
        "original_task_index": 0,
        "init_state_index": 0,
        "task_description": "move cup",
        "policy_seed": 195,
        "init_state_sha256": "a" * 64,
        "query_interval": 1,
        "max_policy_steps": 520,
        "intervention_event_count": 0,
        "intervention_events": [],
        "executed_actions": [
            {"policy_step": i, "action": [0.01 * i, 0, 0, 0, 0, 0, -1]}
            for i in range(1, 5)
        ],
        "success": True,
    }
    intervention = copy.deepcopy(control)
    intervention.update(
        arm="intervention",
        intervention_event_count=1,
        intervention_events=[{"cosmos_query_boundary_step": 2}],
        success=False,
    )
    return control, intervention


def test_native_prefix_includes_action_that_triggers_event():
    control, intervention = pair()
    intervention["executed_actions"][2]["action"][0] = -0.9
    assert verify_pair(control, intervention)["prefix_steps"] == 2
    intervention["executed_actions"][1]["action"][0] = -0.9
    with pytest.raises(ValueError, match="executed_native_prefix_mismatch"):
        verify_pair(control, intervention)


@pytest.mark.parametrize(
    "fault",
    [
        "initial",
        "missing",
        "noncontiguous",
        "control_event",
        "event_zero",
        "action_nan",
    ],
)
def test_invalid_pairs_are_never_accepted(fault):
    control, intervention = pair()
    if fault == "initial":
        intervention["init_state_sha256"] = "b" * 64
    elif fault == "missing":
        intervention["executed_actions"] = intervention["executed_actions"][:1]
    elif fault == "noncontiguous":
        intervention["executed_actions"][0]["policy_step"] = 2
    elif fault == "control_event":
        control["intervention_event_count"] = 1
    elif fault == "event_zero":
        intervention["intervention_events"][0]["cosmos_query_boundary_step"] = 0
    else:
        intervention["executed_actions"][0]["action"][0] = float("nan")
    with pytest.raises(ValueError):
        verify_pair(control, intervention)


def test_unreached_trigger_retains_valid_paired_task_failure():
    control, intervention = pair()
    control["success"] = intervention["success"] = False
    intervention.update(intervention_event_count=0, intervention_events=[])
    receipt = verify_pair(control, intervention)
    assert receipt["matched"] and not receipt["trigger_reached"]
    assert receipt["prefix_steps"] == 4
    intervention["executed_actions"][-1]["action"][0] = 0.5
    with pytest.raises(ValueError, match="executed_native_prefix_mismatch"):
        verify_pair(control, intervention)


def test_commissioning_selection_and_probe_do_not_change_released_cases():
    cases = []
    for source in ("plus", "pro"):
        for event in sorted(EVENTS):
            for index in range(2):
                case = {
                    "case_id": f"{source}-{event}-{index}",
                    "scenario": {
                        "change_type": event,
                        "scenario_id": f"{source}-{event}-{index}",
                        "trigger": {
                            "type": "on_proximity",
                            "value": "cup",
                            "distance_m": 0.18,
                        },
                        "change": {"operation": "move_object", "entity": "cup"},
                    },
                }
                if source == "pro":
                    case["substrate_variant"] = {"benchmark": "LIBERO-PRO"}
                cases.append(case)
    manifest = {"cases": cases}
    original = copy.deepcopy(manifest)
    for source in ("plus", "pro"):
        selected = commissioning_cases(manifest, source)
        assert len(selected) == 8 and all(source_of(c) == source for c in selected)
        for case in selected:
            probe = event_probe(case)
            assert probe["case_id"].startswith("commission-")
            assert probe["scenario"]["trigger"] == {"type": "fixed_step", "value": 2}
            assert probe["scenario"]["change"] == case["scenario"]["change"]
    assert manifest == original


def test_sensor_reads_retain_backend_disturbances():
    env = Environment.__new__(Environment)
    transformed = {"agentview_image": "corrupted-image"}
    env.wrapped = SimpleNamespace(
        backend=SimpleNamespace(refresh_observation=lambda: transformed)
    )
    assert env.read_sensors() is transformed


def test_libero_retry_amendment_does_not_rewrite_historical_protocol():
    old = without_resource_budgets(
        design("docker.io/library/python@sha256:" + "a" * 64)
    )
    validate(old)
    changed = copy.deepcopy(old)
    changed["model"]["retries"] = 3
    with pytest.raises(ValueError, match="retry_or_timing_protocol"):
        validate(changed)
    changed["protocol_version"] = "2-implementation-3-transport-recovery"
    validate(changed)
    assert old["model"]["retries"] == 0
