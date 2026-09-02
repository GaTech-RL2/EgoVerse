"""Tests for the declarative registry (`egomimic/rldb/embodiment/registry/`).

Two things are being defended here. First, that the YAML actually describes the
embodiments the code has: a registry the code does not agree with is worse than
the dicts it replaces. Second, that a malformed block fails at load — a typo'd
field silently ignored is exactly how four sources of truth drifted apart.
"""

import importlib

import pytest

from egomimic.rldb.embodiment import EMBODIMENT, EMBODIMENT_CLASSES, Embodiment
from egomimic.rldb.embodiment.registry import (
    ACTION_SPACES,
    RegistryError,
    _parse_end_effector,
    _parse_platform,
    load_embodiment_platforms,
    load_end_effectors,
    load_platforms,
)


def _import_path(path: str):
    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def test_every_enum_member_is_owned_by_a_platform() -> None:
    by_embodiment = load_embodiment_platforms()
    missing = sorted(
        m.name.lower() for m in EMBODIMENT if m.name.lower() not in by_embodiment
    )
    assert not missing, f"EMBODIMENT members no platform claims: {missing}"


def test_platforms_claim_no_embodiment_that_does_not_exist() -> None:
    known = {m.name.lower() for m in EMBODIMENT}
    for platform in load_platforms().values():
        extra = sorted(set(platform.embodiments) - known)
        assert not extra, f"{platform.name} claims non-existent embodiments: {extra}"


def test_embodiment_class_escape_hatch_resolves() -> None:
    """`embodiment_class:` is a dotted path; a typo must not survive to runtime."""
    for platform in load_platforms().values():
        if platform.embodiment_class is None:
            continue
        cls = _import_path(platform.embodiment_class)
        assert issubclass(cls, Embodiment), platform.embodiment_class


def test_registry_agrees_with_the_class_dict() -> None:
    """Both halves of the registry must name the same class for an embodiment."""
    for embodiment, platform in load_embodiment_platforms().items():
        if embodiment not in EMBODIMENT_CLASSES or platform.embodiment_class is None:
            continue
        assert EMBODIMENT_CLASSES[embodiment] is _import_path(platform.embodiment_class)


def test_default_end_effectors_and_action_spaces_exist() -> None:
    end_effectors = load_end_effectors()
    for platform in load_platforms().values():
        assert platform.default_end_effector in end_effectors
    for spec in end_effectors.values():
        assert spec.action_space in ACTION_SPACES


def test_declared_end_effector_masks() -> None:
    end_effectors = load_end_effectors()
    assert end_effectors["mano_hand"].keypoints.is_complete
    # A jaw is exactly representable: wrist + two tips, aperture = ||tip - tip||.
    jaw = end_effectors["eva_parallel_jaw"].keypoints
    assert jaw.valid == (0, 4, 8)
    assert not jaw.is_complete
    assert jaw.n_slots == 21


def test_aliases_reach_the_same_platform_as_their_target() -> None:
    by_embodiment = load_embodiment_platforms()
    assert by_embodiment["aria_bimanual"] is by_embodiment["human_bimanual"]


def _jaw_block(**overrides):
    block = {
        "class": "parallel_jaw",
        "dof": 1,
        "action_space": "cartesian",
        "keypoints": {"topology": "mano21", "valid": [0, 4, 8]},
    }
    block.update(overrides)
    return block


def _platform_block(**overrides):
    block = {
        "kind": "robot",
        "embodiment_prefix": "eva",
        "arity": ["bimanual"],
        "arm_dof": 6,
        "aux": None,
        "cameras": ["front_1"],
        "reference_frame": "camera:front_1",
        "default_end_effector": "eva_parallel_jaw",
    }
    block.update(overrides)
    return block


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"actoin_space": "cartesian"}, id="typo'd field"),
        pytest.param({"action_space": "joints"}, id="unknown action space"),
        pytest.param({"class": "vacuum_gripper"}, id="unknown class"),
        pytest.param(
            {"keypoints": {"topology": "mano21", "valid": [0, 21]}},
            id="slot out of range",
        ),
        pytest.param(
            {"keypoints": {"topology": "smplx", "valid": "all"}},
            id="unknown topology",
        ),
        pytest.param({"dof": 2, "joint_names": ["a"]}, id="joint_names vs dof"),
        pytest.param({"dof": 1, "joint_limits": [[1.0, -1.0]]}, id="inverted limit"),
        pytest.param({"dof": 1, "dead_dims": [3]}, id="dead dim out of range"),
    ],
)
def test_malformed_end_effector_is_rejected(overrides) -> None:
    with pytest.raises(RegistryError):
        _parse_end_effector("bad_ee", _jaw_block(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"kind": "cyborg"}, id="unknown kind"),
        pytest.param({"arity": []}, id="empty arity"),
        pytest.param({"cameras": []}, id="no cameras"),
        pytest.param(
            {"reference_frame": "camera:left_wrist"}, id="frame names absent camera"
        ),
        pytest.param({"reference_frame": "table"}, id="unknown reference frame"),
        pytest.param({"default_end_effector": "no_such_hand"}, id="unknown default ee"),
        pytest.param({"arm_dof": 0}, id="non-positive arm_dof"),
        pytest.param({"aux": {"dof": 7, "joint_names": ["a"]}}, id="aux names vs dof"),
    ],
)
def test_malformed_platform_is_rejected(overrides) -> None:
    with pytest.raises(RegistryError):
        _parse_platform(
            "bad_platform", _platform_block(**overrides), load_end_effectors()
        )


def test_valid_all_expands_to_the_whole_topology() -> None:
    spec = _parse_end_effector(
        "five_finger",
        _jaw_block(
            **{
                "class": "dexterous_hand",
                "dof": 20,
                "action_space": "keypoints",
                "keypoints": {"topology": "mano21", "valid": "all"},
            }
        ),
    )
    assert spec.keypoints.valid == tuple(range(21))
    assert spec.keypoints.is_complete
