import numpy as np
import pytest

from astra_reversal.reasoning_learning.evidence import CandidateBatch, training_windows


def batch():
    return CandidateBatch(
        observation_id="obs",
        policy_version="v0",
        rule={"lift": True},
        reference=np.zeros((10, 7)),
    )


def test_judgments_cannot_use_changed_policy_rule_or_reference(tmp_path):
    b = batch()
    b.add("lift", np.ones((10, 7)) * 0.2)
    for field in ("policy_version", "observation_id", "reference_sha256", "rule"):
        altered = b.binding
        altered[field] = "changed"
        with pytest.raises(ValueError):
            b.judge(
                "lift", binding=altered, preference="win", rationale="More clearance"
            )
    b.judge("lift", binding=b.binding, preference="win", rationale="More clearance")
    commands, receipt = b.claim("lift", tmp_path)
    np.testing.assert_array_equal(commands, np.full((10, 7), 0.2, dtype=np.float32))
    assert receipt["physical_success"] is None
    with pytest.raises(ValueError):
        b.claim("native", tmp_path)
    # A fresh process restoring the same batch also cannot execute a second time.
    with pytest.raises(FileExistsError):
        batch().claim("native", tmp_path)


@pytest.mark.parametrize("preference", ["loss", "tie", "uncertain"])
def test_only_clear_predicted_win_can_execute(preference, tmp_path):
    b = batch()
    b.add("edit", np.zeros((10, 7)))
    b.judge("edit", binding=b.binding, preference=preference, rationale="No evidence")
    with pytest.raises(ValueError):
        b.claim("edit", tmp_path)


def rows(n=15):
    return [
        {
            "episode_id": "ep",
            "step": i,
            "observation_id": f"obs{i}",
            "action": [i / 100] * 7,
            "executed": True,
            "evidence": "observed_useful",
            "stage": "correction",
            "preference": "win",
            "event_id": "e0",
            "policy_version": "v0",
        }
        for i in range(n)
    ]


def test_windows_pair_actual_pre_action_observation_with_complete_executed_future():
    windows = training_windows(rows(), horizon=10)
    assert [w["observation_id"] for w in windows] == ["obs0", "obs5"]
    assert windows[1]["actions"][0] == pytest.approx([0.05] * 7)
    assert windows[1]["actions"][-1] == pytest.approx([0.14] * 7)
    assert not training_windows(rows(9), horizon=10)


@pytest.mark.parametrize(
    "field,value",
    [
        ("evidence", "predicted_acceptable"),
        ("evidence", "failed"),
        ("executed", False),
        ("preference", "tie"),
        ("episode_id", "other"),
    ],
)
def test_unverified_failed_or_unexecuted_steps_are_not_imitated(field, value):
    steps = rows()
    steps[7][field] = value
    assert training_windows(steps, horizon=10) == []


def test_useful_native_setup_does_not_need_a_fictitious_comparison():
    steps = rows(10)
    for row in steps:
        row.update(stage="setup", preference=None)
    assert len(training_windows(steps, horizon=10)) == 1


def test_external_mutation_does_not_change_fixed_reference_or_rule():
    reference = np.zeros((10, 7))
    rule = {"lift": True}
    b = CandidateBatch(
        observation_id="o", policy_version="v", rule=rule, reference=reference
    )
    reference[:] = 1
    rule["lift"] = False
    b.proposals()["native"][:] = 2
    assert not b.proposals()["native"].any()
    assert b.binding["rule"] == {"lift": True}
