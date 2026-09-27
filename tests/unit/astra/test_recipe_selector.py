import json

import numpy as np
import pytest
import torch

from astra_reversal.recipe_selector import (
    PhaseSelector,
    canonical_choice,
    pooled_features,
)


def choice(operator="tli", alpha=0.25):
    return {
        "operator": operator,
        "source_a_id": "10",
        "source_b_id": "14",
        "alpha": alpha,
    }


def test_equivalent_pair_swap_preserves_both_operator_formulas():
    a, b = np.array([1.0, 3.0]), np.array([-4.0, 2.0])
    for operator in ("tei", "tli"):
        original = {**choice(operator, 0.75), "source_a_id": "14", "source_b_id": "10"}
        normalized = canonical_choice(original)
        assert normalized == choice(operator, 0.25)
        if operator == "tei":
            assert np.array_equal(0.25 * b + 0.75 * a, 0.75 * a + 0.25 * b)
        else:
            assert np.array_equal(-0.5 * (b - a), 0.5 * (a - b))
    assert canonical_choice(choice(alpha=0.5)) == {"operator": "native"}
    assert canonical_choice(choice("tei", 0.5))["operator"] == "tei"


def test_pooling_excludes_padding_and_protected_text_slots():
    hidden = np.arange(14, dtype=np.float32).reshape(1, 7, 2)
    padding = np.array([[True, True, False, True, True, True, False]])
    instruction = np.array([[False, True, True, False]])
    state = np.arange(8, dtype=np.float32)
    result = pooled_features(hidden, padding, instruction, state)
    assert np.array_equal(
        result, np.r_[hidden[0, :2].mean(0), hidden[0, 4:6].mean(0), state]
    )
    hidden[:, 2] = 100000
    hidden[:, 3] = -100000
    hidden[:, 6] = 100000
    assert np.array_equal(result, pooled_features(hidden, padding, instruction, state))
    instruction[0, 3] = True
    with pytest.raises(ValueError, match="masks"):
        pooled_features(hidden, padding, instruction, state)


def test_selector_training_is_observation_conditioned_and_reloads(tmp_path):
    torch.set_num_threads(1)
    features = np.array(
        [[-3.0, 0], [-2.0, 0], [2.0, 3], [3.0, 3], [2.0, -3], [3.0, -3]], np.float32
    )
    other = {"operator": "tei", "source_a_id": "13", "source_b_id": "18", "alpha": 0.8}
    choices = [
        {"operator": "native"},
        {"operator": "native"},
        choice(),
        choice(),
        other,
        other,
    ]
    groups = ["a", "a", "b", "b", "c", "c"]
    selector = PhaseSelector.for_choices(features, choices, device="cpu")
    with pytest.raises(RuntimeError, match="allocated GPU"):
        selector.fit(features, choices, groups, source_sha256="a" * 64)
    receipt = selector.fit(
        features, choices, groups, source_sha256="a" * 64, _test_steps=200
    )
    assert receipt["optimizer_steps"] == 200 and receipt["test_only"]
    assert selector.predict(features[0])["choice"] == {"operator": "native"}
    assert selector.predict(features[2])["choice"]["operator"] == "tli"
    assert abs(selector.predict(features[2])["choice"]["alpha"] - 0.25) < 0.05
    assert selector.predict(features[-1])["choice"]["operator"] == "tei"
    assert selector.predict(features[-1])["choice"]["source_a_id"] == "13"
    assert abs(selector.predict(features[-1])["choice"]["alpha"] - 0.8) < 0.05
    root = tmp_path / "selector"
    selector.save(root)
    with pytest.raises(ValueError, match="training scope"):
        PhaseSelector.load(root, device="cpu")
    restored = PhaseSelector.load(root, device="cpu", allow_test_only=True)
    assert restored.predict(features[-1]) == selector.predict(features[-1])
    with pytest.raises(ValueError, match="refit"):
        selector.fit(features, choices, groups, source_sha256="a" * 64, _test_steps=1)
    (root / "weights.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes"):
        PhaseSelector.load(root, device="cpu", allow_test_only=True)
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["files"].pop("weights.pt")
    (root / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="inventory"):
        PhaseSelector.load(root, device="cpu", allow_test_only=True)


def test_gate_tie_defers_and_failed_input_is_not_accepted():
    selector = PhaseSelector(2, [("tei", "10", "14")], device="cpu")
    with torch.no_grad():
        selector.model.gate.weight.zero_()
        selector.model.gate.bias.zero_()
    selector.training = {"test_only": True}
    decision = selector.predict(np.array([1, 2], np.float32))
    assert decision["gate_probability"] == 0.5
    assert decision["choice"] == {"operator": "native"}
    with pytest.raises(ValueError, match="dimension or values"):
        selector.predict(np.array([np.nan, 0]))
    with torch.no_grad():
        selector.model.gate.bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        selector.predict(np.array([1, 2], np.float32))


def test_config_pins_test_denominators_and_provider_exclusion():
    from pathlib import Path

    config = json.loads(
        (
            Path(__file__).parents[3]
            / "astra_reversal/configs/learned_correction_recipe_v1.json"
        ).read_text()
    )
    evaluation = config["evaluation"]
    ood = (
        len(evaluation["ood_suites"])
        * len(evaluation["ood_task_ids"])
        * len(evaluation["ood_reset_ids"])
    )
    id_count = len(evaluation["id_task_ids"]) * len(evaluation["id_reset_ids"])
    assert (ood + id_count) * len(config["arms"]) == 240
    assert config["provider_calls_allowed"] is False
    assert 0 not in evaluation["id_reset_ids"]
    assert 0 not in evaluation["ood_reset_ids"]
