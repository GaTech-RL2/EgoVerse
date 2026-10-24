"""dexmate_bimanual is a valid embodiment with its own id; the existing ids are
unchanged (norm-stat caches and precomputed stats are keyed by them)."""

from egomimic.rldb.embodiment.embodiment import (
    EMBODIMENT,
    get_embodiment,
    get_embodiment_id,
)
from egomimic.rldb.zarr import dexmate_rules
from egomimic.rldb.zarr.episode_rules import EMBODIMENTS, arm_sides, embodiment_error


def test_existing_ids_are_unchanged_and_dexmate_is_8():
    assert {e.name: e.value for e in EMBODIMENT} == {
        "HUMAN_RIGHT_ARM": 1,
        "HUMAN_LEFT_ARM": 2,
        "HUMAN_BIMANUAL": 3,
        "EVA_RIGHT_ARM": 4,
        "EVA_LEFT_ARM": 5,
        "EVA_BIMANUAL": 6,
        "DEXMATE_BIMANUAL": 8,
    }


def test_dexmate_name_round_trips():
    assert dexmate_rules.EMBODIMENT == "dexmate_bimanual"
    assert dexmate_rules.EMBODIMENT in EMBODIMENTS
    assert embodiment_error("dexmate_bimanual") is None
    assert get_embodiment_id("dexmate_bimanual") == 8
    assert get_embodiment(8) == "DEXMATE_BIMANUAL"
    assert arm_sides("dexmate_bimanual") == dexmate_rules.SIDES
