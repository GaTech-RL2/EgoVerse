import numpy as np
import pytest

from astra_reversal.intervention_search import arm_summary, load_protocol
from astra_reversal.interventions import (
    apply_vision,
    noise_basis,
    perturb_noise,
    success_curve,
)
from astra_reversal.osmo.interventions import assignment


def test_full_latent_noise_is_bounded_and_replayable():
    shape = (1, 10, 32)
    basis, identity = noise_basis(shape, 19)
    repeated, repeated_id = noise_basis(shape, 19)
    np.testing.assert_array_equal(basis, repeated)
    assert identity == repeated_id
    proposal = {
        "basis_id": identity,
        "coefficients": [1.0] + [0.0] * 7,
        "perturbation_scale": 0.5,
    }
    original = np.ones(shape, dtype=np.float32)
    changed = perturb_noise(original, basis, proposal)
    assert np.sqrt(np.mean((changed - original) ** 2)) == pytest.approx(0.5)
    assert np.any(changed[..., 7:] != original[..., 7:])
    np.testing.assert_array_equal(original, np.ones(shape))
    with pytest.raises(ValueError, match="different basis"):
        perturb_noise(original, basis, {**proposal, "basis_id": "wrong"})
    with pytest.raises(ValueError, match="Invalid bounded"):
        perturb_noise(original, basis, {**proposal, "coefficients": [1.0] * 8})


def test_vision_does_not_change_feedback_geometry_or_proprioception():
    observation = {
        "observation/image": np.zeros((12, 16, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((12, 16, 3), dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
    }
    annotation = {
        "camera": "observation/image",
        "kind": "box",
        "coordinates": [2, 3, 9, 10],
        "gain": 0.5,
    }
    changed = apply_vision(observation, [annotation])
    assert changed["observation/image"].shape == observation["observation/image"].shape
    assert changed["observation/image"].sum() > 0
    assert observation["observation/image"].sum() == 0
    np.testing.assert_array_equal(
        changed["observation/state"], observation["observation/state"]
    )
    np.testing.assert_array_equal(
        changed["observation/wrist_image"], observation["observation/wrist_image"]
    )
    unchanged = apply_vision(observation, [{**annotation, "gain": 0}])
    np.testing.assert_array_equal(
        unchanged["observation/image"], observation["observation/image"]
    )


def test_failed_provider_call_consumes_attempt_without_rollout_or_fake_success():
    attempts = [
        {
            "iteration": 1,
            "success": False,
            "rollout_executed": True,
            "status": "budget_exhausted",
            "actions_executed": 300,
            "velocity_evaluations": 600,
            "wall_seconds": 8,
        },
        {
            "iteration": 2,
            "success": False,
            "rollout_executed": False,
            "status": "proposal_error",
            "actions_executed": 0,
            "velocity_evaluations": 0,
            "wall_seconds": 0,
        },
        {
            "iteration": 3,
            "success": True,
            "rollout_executed": True,
            "status": "success",
            "actions_executed": 100,
            "velocity_evaluations": 200,
            "wall_seconds": 3,
        },
    ]
    report = arm_summary(attempts, 5)
    assert report["first_success_attempt"] == 3
    assert report["intervention_iterations_to_success"] == 2
    assert report["success_by_attempt"] == [False, False, True, True, True]
    assert report["candidate_rollouts"] == 2
    assert report["proposal_failures"] == 1
    assert report["actions_through_success_or_cap"] == 400
    failed = success_curve(attempts[:2], 5)
    assert failed["first_success_attempt"] is None
    assert failed["censored_after_attempt"] == 5


def test_prespecified_shards_cover_all_cases_once():
    protocol = load_protocol()
    assigned = []
    for worker in range(8):
        target = assignment("evaluation", worker)
        assigned += [
            (target["suite"], *case)
            for case in protocol["evaluation_cases"][
                target["case_shard"] :: target["case_shards"]
            ]
        ]
    assert len(assigned) == len(set(assigned)) == 20
    assert len(protocol["arms"]) == 8
    with pytest.raises(ValueError):
        assignment("evaluation", 8)
