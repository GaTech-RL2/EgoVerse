"""The evaluator must actually evaluate under the controller it names.

This is the study's dependent variable: `unseen_jittery` SR is only evidence
about controller generalization if the jittery gap is really applied. A gap
that silently stayed `ideal` would produce six identical, plausible-looking
numbers and no error anywhere.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymunk")

from Tsimulation.pushshapes.agents import CONTROL_GAPS  # noqa: E402
from egomimic.eval.core.eval_sim import SimRolloutEval  # noqa: E402

GRIPPER = dict(
    embodiment_name="pushshapes_sim_gripper",
    env_kwargs={"object_shape": "T", "pusher_shape": "gripper",
                "obstacle_level": 0, "image_size": 96},
    init_mode="seeds",
    init_seeds=[0, 1],
    max_steps=10,
)


def test_named_gap_reaches_the_agent():
    ev = SimRolloutEval(**GRIPPER, control_gap="jittery")
    env = ev._get_env()
    assert env.agent.control_gap.as_dict() == CONTROL_GAPS["jittery"].as_dict()
    assert env.agent.randomize_gap is False


def test_gap_survives_reset():
    """`reset()` calls `reset_control_gap`; it must not revert to ideal."""
    ev = SimRolloutEval(**GRIPPER, control_gap="sticky")
    env = ev._get_env()
    env.reset(seed=3)
    assert env.agent.control_gap.as_dict() == CONTROL_GAPS["sticky"].as_dict()
    env.reset(seed=11)
    assert env.agent.control_gap.as_dict() == CONTROL_GAPS["sticky"].as_dict()


def test_default_is_ideal_and_unchanged():
    """Omitting control_gap must leave existing evaluators bit-identical."""
    ev = SimRolloutEval(**GRIPPER)
    assert ev.control_gap is None
    env = ev._get_env()
    assert env.agent.control_gap.as_dict() == CONTROL_GAPS["ideal"].as_dict()


def test_every_studied_gap_is_distinct_from_ideal():
    """Guards the experiment itself: five seen modes and one held-out mode
    that all collapsed onto `ideal` would compare a model against itself."""
    ideal = CONTROL_GAPS["ideal"].as_dict()
    for mode in ("tight", "loose", "laggy", "sticky", "jittery"):
        env = SimRolloutEval(**GRIPPER, control_gap=mode)._get_env()
        assert env.agent.control_gap.as_dict() != ideal, f"{mode} == ideal"


def test_unknown_gap_fails_at_construction_not_on_the_node():
    with pytest.raises(ValueError, match="unknown control_gap"):
        SimRolloutEval(**GRIPPER, control_gap="jittry")


def test_control_gap_cannot_be_smuggled_through_env_kwargs():
    """Documents WHY the parameter exists: env_kwargs cannot carry it."""
    bad = dict(GRIPPER)
    bad["env_kwargs"] = dict(bad["env_kwargs"], control_gap="jittery")
    with pytest.raises(TypeError, match="unexpected PushShapesEnv option"):
        SimRolloutEval(**bad)._get_env()
