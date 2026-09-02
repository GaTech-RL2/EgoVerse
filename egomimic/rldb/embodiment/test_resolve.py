"""Tests for `Embodiment.resolve` — the composition path.

The property under test is the design's acceptance test: swapping the
end-effector must change `action_space` and nothing else, and swapping the
platform must change the platform and nothing about the hand.
"""

from dataclasses import replace

import pytest

from egomimic.rldb.embodiment import Embodiment, Eva, Human, ResolvedEmbodiment
from egomimic.rldb.embodiment.registry import load_end_effectors, load_platforms


def test_resolve_by_name_uses_the_platform_default_end_effector() -> None:
    """No episode carries a `morphology` block yet, so the name must still work."""
    eva = Embodiment.resolve("eva_bimanual")
    assert eva.platform is load_platforms()["eva_x5"]
    assert eva.platform.arm_dof == 6
    assert set(eva.end_effectors) == {"left", "right"}
    assert all(ee.name == "eva_parallel_jaw" for ee in eva.end_effectors.values())
    assert eva.action_space == "cartesian"
    assert eva.embodiment_class is Eva


def test_resolve_human_lands_on_the_keypoints_head() -> None:
    human = Embodiment.resolve("human_bimanual")
    assert human.platform is load_platforms()["human_body"]
    assert human.platform.arm_dof is None
    assert human.action_space == "keypoints"
    assert human.embodiment_class is Human
    assert human.keypoints("left").is_complete


def test_resolve_follows_aliases() -> None:
    assert (
        Embodiment.resolve("aria_bimanual").platform
        is Embodiment.resolve("human_bimanual").platform
    )
    assert Embodiment.resolve("SCALE_LEFT_ARM").embodiment_class is Human


def test_resolve_from_morphology_block() -> None:
    resolved = Embodiment.resolve(
        {
            "platform": "eva_x5",
            "end_effector": {"left": "eva_parallel_jaw", "right": "eva_parallel_jaw"},
            "vendor": "rl2",
        }
    )
    assert resolved.platform.name == "eva_x5"
    assert resolved.action_space == "cartesian"
    assert resolved.embodiment_class is Eva


def test_morphology_end_effector_may_be_a_bare_string() -> None:
    resolved = Embodiment.resolve(
        {"platform": "human_body", "end_effector": "mano_hand"}
    )
    assert set(resolved.end_effectors) == {"left", "right"}
    assert resolved.action_space == "keypoints"


def test_a_new_hand_on_a_known_platform_costs_no_code() -> None:
    """The design's acceptance test, run against the registry as it stands.

    Swapping the declared end-effector re-routes the head with no change to the
    platform, the stem inputs or any Python. Here the jaw is swapped for a
    five-finger hand on the same EVA arms.
    """
    jaws = Embodiment.resolve(
        {"platform": "eva_x5", "end_effector": "eva_parallel_jaw"}
    )
    hands = Embodiment.resolve({"platform": "eva_x5", "end_effector": "mano_hand"})

    assert jaws.platform is hands.platform
    assert jaws.platform.cameras == hands.platform.cameras
    assert jaws.platform.arm_dof == hands.platform.arm_dof
    assert jaws.action_space == "cartesian"
    assert hands.action_space == "keypoints"


def test_mixed_action_spaces_are_refused() -> None:
    """Two hands wanting different heads has no single training interface."""
    mixed = Embodiment.resolve(
        {
            "platform": "eva_x5",
            "end_effector": {"left": "eva_parallel_jaw", "right": "mano_hand"},
        }
    )
    with pytest.raises(ValueError, match="disagree on action_space"):
        mixed.action_space


def test_platform_without_a_class_says_so() -> None:
    platform = replace(load_platforms()["eva_x5"], embodiment_class=None)
    resolved = ResolvedEmbodiment(
        platform=platform,
        end_effectors={"left": load_end_effectors()["eva_parallel_jaw"]},
    )
    with pytest.raises(NotImplementedError, match="embodiment_class"):
        resolved.embodiment_class


@pytest.mark.parametrize(
    "spec, match",
    [
        pytest.param("not_an_embodiment", "not owned by any platform", id="bad name"),
        pytest.param({"platform": "nope"}, "not in", id="bad platform"),
        pytest.param(
            {"platform": "eva_x5", "end_effector": {"left": "nope"}},
            "not in",
            id="bad end effector",
        ),
        pytest.param(
            {"platform": "eva_x5", "end_effector": {"middle": "mano_hand"}},
            "unknown side",
            id="bad side",
        ),
        pytest.param(
            {"platform": "eva_x5", "end_effector": []}, "must be", id="bad ee type"
        ),
    ],
)
def test_a_bad_identifier_is_a_hard_error(spec, match) -> None:
    """Identifiers plus a registry: a typo fails at resolve, not silently later."""
    with pytest.raises(ValueError, match=match):
        Embodiment.resolve(spec)


def test_resolve_rejects_the_wrong_type() -> None:
    with pytest.raises(TypeError):
        Embodiment.resolve(3)


def test_resolve_delegates_to_the_hand_written_pipeline() -> None:
    """`Human`/`Eva` are reached through the registry, not rewritten."""
    eva = Embodiment.resolve("eva_bimanual")
    assert eva.get_keymap("cartesian") == Eva.get_keymap("cartesian")
    assert len(eva.get_transform_list("cartesian")) == len(
        Eva.get_transform_list("cartesian")
    )
