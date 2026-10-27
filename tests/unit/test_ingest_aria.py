"""Aria uploads through ingest (egomimic/ingest/aria.py), up to the conversion.

Converting needs a real VRS and its MPS outputs: tests/integration/test_ingest_aria.py.
"""

import json

import pytest

from egomimic.ingest import aria
from egomimic.ingest.formats import detect, prepare_delivery
from egomimic.rldb.lance.schema import recording_id

VID = "1758390171000"
MPS = [
    f"mps_{VID}_vrs/hand_tracking/hand_tracking_results.csv",
    f"mps_{VID}_vrs/slam/closed_loop_trajectory.csv",
]
# A rig's metadata file as it is written today (raw_v2/aria/): embodiment is the device,
# robot_name the arm layout; no rig_name, device_serial or license.
RIG_FILE = {
    "embodiment": "aria",
    "episode_hash": int(VID),
    "operator": "op_7",
    "lab": "rl2",
    "task": "fold clothes",
    "robot_name": "aria_bimanual",
    "task_description": "fold the shirts",
    "scene": "1",
    "objects": ["shirt"],
}


def upload(tmp_path, meta=RIG_FILE, mps_rows=True, vid=VID):
    d = tmp_path / "delivery"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{vid}.vrs").write_bytes(b"not a real vrs")
    (d / f"{vid}_metadata.json").write_text(json.dumps(meta))
    for rel in MPS:
        p = d / rel.replace(VID, vid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("header\n" + ("1,2,3\n" if mps_rows else ""))
    (d / f"mps_{vid}_vrs" / "eye_gaze.zip").write_bytes(b"zip")
    return d


def prepare(tmp_path, d, **kw):
    return prepare_delivery(
        d, tmp_path / "staged", tmp_path / "lance", fingerprint="f" * 64,
        vendor="rl2-glasses", mano=False, **kw,
    )  # fmt: skip


def test_an_aria_upload_is_detected_by_its_files(tmp_path):
    d = upload(tmp_path)
    names = [p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()]
    fmt = detect(names)
    assert fmt.name == "aria" and fmt.recording_key(names) == VID
    assert fmt.unexpected(set(names)) == []  # the rest of the MPS folder is kept
    assert detect([f"{VID}.vrs", f"{VID}_metadata.json"]) is None  # no MPS


@pytest.mark.parametrize(
    "layout, embodiment",
    [
        ("aria_bimanual", "human_bimanual"),
        ("aria_left_arm", "human_left_arm"),
        ("aria right arm", "human_right_arm"),
        ("human_bimanual", "human_bimanual"),
    ],
)
def test_the_rigs_arm_layout_is_the_human_embodiment(layout, embodiment):
    meta = aria.upload_metadata(RIG_FILE | {"robot_name": layout}, VID, "CC-BY-4.0")
    assert (meta["embodiment"], meta["rig_name"]) == (embodiment, "aria")
    assert (meta["task_name"], meta["license"]) == ("fold clothes", "CC-BY-4.0")


def test_the_misspelled_robot_name_key_is_read():
    md = {k: v for k, v in RIG_FILE.items() if k != "robot_name"}
    meta = aria.upload_metadata(md | {"robot name": "aria_left_arm"}, VID)
    assert meta["embodiment"] == "human_left_arm"


def test_objects_none_is_a_waiver_for_review():
    meta = aria.upload_metadata(RIG_FILE | {"objects": ["None"]}, VID)
    assert "objects" not in meta and meta["missing_metadata"] == ["objects"]


def test_a_rig_file_needs_only_a_license_to_reach_the_conversion(tmp_path):
    p = prepare(tmp_path, upload(tmp_path))
    assert p.status == "rejected" and p.recording_id == recording_id("aria", VID)
    assert p.problems == [
        "license is required, got None (set it in the metadata, or give the vendor a "
        "default at onboarding (onboard --license))"
    ]
    # With the vendor's license it gets past every check; the fake VRS then fails to
    # convert (as a corrupt upload would), which is "failed", not "rejected".
    p = prepare(tmp_path, upload(tmp_path), default_license="CC-BY-4.0")
    assert p.status == "failed" and p.problems == []


def test_mps_without_rows_is_rejected_before_any_gpu_time(tmp_path):
    p = prepare(tmp_path, upload(tmp_path, mps_rows=False), default_license="CC-BY-4.0")
    assert p.status == "rejected"
    assert p.problems == [f"{rel} has no rows" for rel in MPS]


def test_an_eva_layout_is_not_an_aria_recording(tmp_path):
    meta = RIG_FILE | {"robot_name": "eva_bimanual"}
    p = prepare(tmp_path, upload(tmp_path, meta=meta), default_license="CC-BY-4.0")
    assert p.status == "rejected"
    assert p.problems[0].startswith("embodiment 'eva_bimanual' is not one of")


def test_an_id_that_is_not_epoch_ms_is_rejected(tmp_path):
    p = prepare(tmp_path, upload(tmp_path, vid="1758390171"), default_license="X")
    assert p.status == "rejected" and "epoch milliseconds" in p.problems[0]


def test_an_unreadable_vrs_has_no_serial(tmp_path):
    (tmp_path / "x.vrs").write_bytes(b"junk")
    assert aria.vrs_serial(tmp_path / "x.vrs") is None
