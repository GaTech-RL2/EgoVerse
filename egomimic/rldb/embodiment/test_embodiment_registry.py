"""Tests for the single embodiment registry (`egomimic.rldb.embodiment`)."""

import pytest

from egomimic.rldb.embodiment import (
    EMBODIMENT,
    EMBODIMENT_CLASSES,
    Embodiment,
    Human,
    canonical_embodiment_name,
    get_embodiment_class,
    get_embodiment_id,
)
from egomimic.rldb.embodiment.registry import load_aliases


def test_every_enum_member_has_a_class() -> None:
    """The registry is the single answer to "what embodiments are there".

    An enum member with no class is the drift the two mirrored dicts used to
    hide: the name loads, the overlay silently falls back to `Human`.
    """
    missing = [
        m.name.lower() for m in EMBODIMENT if m.name.lower() not in EMBODIMENT_CLASSES
    ]
    assert not missing, f"EMBODIMENT members with no class: {missing}"


def test_registry_has_no_unknown_names() -> None:
    known = {m.name.lower() for m in EMBODIMENT}
    extra = sorted(set(EMBODIMENT_CLASSES) - known)
    assert not extra, f"EMBODIMENT_CLASSES names not in the EMBODIMENT enum: {extra}"


def test_every_class_is_an_embodiment() -> None:
    for name, cls in EMBODIMENT_CLASSES.items():
        assert issubclass(cls, Embodiment), f"{name} -> {cls} is not an Embodiment"


def test_get_embodiment_class_is_case_insensitive() -> None:
    assert get_embodiment_class("EVA_BIMANUAL") is EMBODIMENT_CLASSES["eva_bimanual"]
    assert get_embodiment_class("Human_Bimanual") is Human


def test_get_embodiment_class_falls_back_instead_of_raising() -> None:
    """Readers pull this name out of an episode's attrs; a miss is a fallback."""
    assert get_embodiment_class("not_an_embodiment") is None
    assert get_embodiment_class("") is None
    assert get_embodiment_class(None) is None
    assert get_embodiment_class("not_an_embodiment", default=Human) is Human


def test_aliases_resolve_to_live_ids() -> None:
    """The 07/08/2026 collapse hard-crashed on cached episodes; it must not now."""
    for old, new in load_aliases().items():
        assert get_embodiment_id(old) == get_embodiment_id(new), old


def test_alias_targets_are_canonical_names() -> None:
    known = {m.name.lower() for m in EMBODIMENT}
    aliases = load_aliases()
    unknown = sorted(t for t in aliases.values() if t not in known)
    assert not unknown, f"aliases.yaml points at names that do not exist: {unknown}"


def test_no_alias_shadows_a_live_name() -> None:
    """A live name always wins, so a re-used name can never be silently rerouted."""
    known = {m.name.lower() for m in EMBODIMENT}
    shadowing = sorted(set(load_aliases()) & known)
    assert not shadowing, f"aliases.yaml keys collide with live names: {shadowing}"
    for name in known:
        assert canonical_embodiment_name(name) == name


def test_aliases_are_not_chained() -> None:
    """One hop only: an alias target must not itself be an alias."""
    aliases = load_aliases()
    chained = sorted(t for t in aliases.values() if t in aliases)
    assert not chained, f"aliases.yaml has chained entries: {chained}"


def test_alias_resolves_to_the_right_class() -> None:
    assert get_embodiment_class("aria_bimanual") is Human
    assert get_embodiment_class("MECKA_LEFT_ARM") is Human


def test_unknown_name_still_fails_loudly() -> None:
    """An alias table is a compatibility shim, not a way to swallow typos."""
    with pytest.raises(KeyError):
        get_embodiment_id("eva_bimanualll")


def test_aliases_are_a_read_shim_only(tmp_path) -> None:
    """Old episodes load under an alias; new ones may not be *written* under one.

    The table exists so a rename costs nothing at load. Letting a producer write
    a deprecated name would turn the shim into a second live spelling.
    """
    import numpy as np

    from egomimic.rldb.zarr.zarr_writer import ZarrWriter

    with pytest.raises(ValueError, match="embodiment must be one of"):
        ZarrWriter.create_and_write(
            episode_path=tmp_path / "aliased.zarr",
            numeric_data={"left.obs_gripper": np.zeros((4, 1))},
            embodiment="aria_bimanual",
            intrinsics={"front_1": np.zeros((3, 4))},
        )
