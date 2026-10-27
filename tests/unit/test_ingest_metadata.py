"""Upload metadata shared by every ingest format (egomimic/ingest/metadata.py)."""

import pytest

from egomimic.ingest.metadata import check_metadata, waived_fields

EVA = ("eva_bimanual", "eva_left_arm", "eva_right_arm")
META = {
    "task_name": "fold_clothes",
    "task_description": "fold the shirt",
    "operator": "op_7",
    "lab": "rl2",
    "scene": "kitchen_A",
    "objects": ["shirt", "basket"],
    "embodiment": "eva_bimanual",
    "rig_name": "eva_2",
    "device_serial": "SN0001",
    "license": "CC-BY-4.0",
    "vendor_recording_id": "1760000000000",
}


def test_complete_metadata_passes():
    assert check_metadata(META, EVA) == ([], [])


def test_waivable_fields_go_through_as_warnings():
    meta = {k: v for k, v in META.items() if k != "scene"} | {
        "missing_metadata": ["scene"]
    }
    assert waived_fields(meta) == {"scene"}
    problems, warnings = check_metadata(meta, EVA)
    assert problems == [] and warnings == [
        "scene: marked as not available (MissingMetadata)"
    ]
    problems, _ = check_metadata(meta, EVA, allow_missing=False)
    assert "--allow-missing" in problems[0]


@pytest.mark.parametrize("name", ["lab", "task_name", "operator"])
def test_unwaivable_fields_are_problems(name):
    meta = {k: v for k, v in META.items() if k != name} | {"missing_metadata": name}
    problems, _ = check_metadata(meta, EVA)
    assert any("cannot be waived" in p for p in problems)


def test_a_missing_or_null_key_is_a_problem_not_a_silent_null():
    problems, _ = check_metadata({**META, "scene": None}, EVA)
    assert problems  # null without a waiver
    problems, _ = check_metadata(
        {k: v for k, v in META.items() if k != "device_serial"}, EVA
    )
    assert problems == ["device_serial is required, got None"]


def test_placeholders_and_unknown_embodiments_are_refused():
    problems, _ = check_metadata({**META, "operator": "unknown"}, EVA)
    assert "placeholder" in problems[0]
    problems, _ = check_metadata({**META, "embodiment": "human_bimanual"}, EVA)
    assert problems == [
        "embodiment 'human_bimanual' is not one of "
        "['eva_bimanual', 'eva_left_arm', 'eva_right_arm']"
    ]
    problems, _ = check_metadata({**META, "vendor_recording_id": "a/b"}, EVA)
    assert "single path component" in problems[0]
