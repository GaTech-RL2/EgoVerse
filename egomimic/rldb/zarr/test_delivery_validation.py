"""Complete ego delivery guarantees and episode-local malformed-input reports."""

import json

import numpy as np
import pytest
import zarr

from egomimic.rldb.embodiment import Embodiment
from egomimic.rldb.zarr.test_dexterous import write_dexterous as write_dexterous
from egomimic.rldb.zarr.test_render_check import overlay_episode as overlay_episode
from egomimic.rldb.zarr.validate import main, validate_episode, waivable_rules
from egomimic.rldb.zarr.zarr_writer import ZarrWriter


@pytest.mark.parametrize("key,value", [
    ("total_frames", "four"), ("total_frames", -2), ("total_frames", True),
    ("features", ["images.front_1"]),
    ("features", {"images.front_1": "jpeg"}),
    ("features", {"images.front_1": {"shape": "64x64"}}),
    ("embodiment", {"name": "human_bimanual"}),
    ("extrinsics", {"left": np.zeros((4, 4)).tolist()}),
])
def test_bad_metadata_is_reported_and_next_cli_episode_is_checked(overlay_episode, tmp_path, key, value, capsys):
    group = zarr.open_group(overlay_episode, mode="a")
    group.attrs[key] = value
    assert not validate_episode(overlay_episode).ok
    assert main([str(overlay_episode), str(tmp_path / "absent.zarr"), "--json"]) == 1
    reports = json.loads(capsys.readouterr().out)
    assert len(reports) == 2 and not any(r["ok"] for r in reports)


@pytest.mark.parametrize("key,column,value", [
    ("left.obs_ee_pose", 0, np.nan), ("left.obs_ee_pose", 3, 0),
    ("obs_head_pose", 3, 0), ("obs_head_pose", 0, np.inf),
])
def test_retained_invalid_poses_fail_but_padding_is_ignored(overlay_episode, key, column, value):
    group = zarr.open_group(overlay_episode, mode="a")
    group[key][4:, column] = value
    assert validate_episode(overlay_episode).ok
    group[key][2, column] = value
    report = validate_episode(overlay_episode)
    assert any(f.check == key for f in report.errors), report.text()


def test_human_missing_estimates_are_limited_not_corrupt(overlay_episode):
    group = zarr.open_group(overlay_episode, mode="a")
    group["left.obs_ee_pose"][0] = np.full(7, 1e9)
    group["left.obs_keypoints"][0] = np.full(63, np.nan)
    report = validate_episode(overlay_episode)
    assert report.ok, report.text()
    assert any("missing human" in f.message for f in report.warnings)
    assert not validate_episode(overlay_episode, ego_overlay=True).ok


@pytest.fixture
def complete_delivery(write_dexterous, tmp_path):
    base = zarr.open_group(write_dexterous(), mode="r")
    numeric = {name: np.asarray(base[name][:]) for name in base.array_keys() if name != "annotations"}
    numeric["obs_head_pose"] = np.tile([0, 0, 0, 1, 0, 0, 0], (6, 1)).astype(float)
    path = tmp_path / "complete.zarr"
    ZarrWriter.create_and_write(path, numeric_data=numeric,
        image_data={"images.front_1": np.zeros((6, 64, 64, 3), np.uint8)},
        embodiment="eva_bimanual", task_name="packing", chunk_timesteps=4,
        calibration={"reference_frame": "robot_base", "cameras": {"front_1": {
            "K": [[50, 0, 32, 0], [0, 50, 32, 0], [0, 0, 1, 0]],
            "model": "PINHOLE", "rectified": True, "resolution": [64, 64],
        }}}, metadata_override={"schema_version": "v3.1", "morphology": {
            "platform": "test_dexterous_platform", "end_effector": "test_hand_v1"}},
        annotations=[("Pack objects", 0, 6)])
    group = zarr.open_group(path, mode="a")
    features = dict(group.attrs["features"])
    resolved = Embodiment.from_attrs(group.attrs)
    for side, spec in resolved.end_effectors.items():
        for prefix in ("obs", "cmd"):
            features[f"{side}.{prefix}_hand_joints"]["joint_names"] = list(spec.joint_names)
    group.attrs["features"] = features
    return path


def test_complete_command_checks_all_retained_rgb_frames_and_reports_status(complete_delivery):
    requirements = {name: True for name in waivable_rules()}
    report = validate_episode(complete_delivery, requirements=requirements, ego_overlay=True)
    assert report.ok, report.text()
    assert report.capabilities["ego_overlay"]["checked_frames"] == 6
    assert report.status_eligible
    group = zarr.open_group(complete_delivery, mode="a")
    group.attrs["data_status"] = "structural_sample"
    sample = validate_episode(complete_delivery, ego_overlay=True)
    assert sample.ok and not sample.status_eligible
    assert sample.to_jsonable()["capabilities"]["ego_overlay"]["available"]
    assert "status eligible=no" in sample.summary()


@pytest.mark.parametrize("change", ["head", "head_tail", "K", "rgb", "resolution", "keypoints", "order", "arm", "nan_joints", "nan_keypoints"])
def test_complete_delivery_exposes_missing_or_invalid_consumer_inputs(complete_delivery, change):
    group = zarr.open_group(complete_delivery, mode="a")
    if change == "head":
        del group["obs_head_pose"]
    elif change == "head_tail":
        group["obs_head_pose"][5, 3] = 0
    elif change == "K":
        block = dict(group.attrs["calibration"])
        del block["cameras"]["front_1"]["K"]
        group.attrs["calibration"] = block
    elif change == "rgb":
        group["images.front_1"][5:6] = np.array([b"invalid jpeg"], dtype=object)
    elif change == "resolution":
        block = dict(group.attrs["calibration"])
        block["cameras"]["front_1"]["resolution"] = [320, 240]
        group.attrs["calibration"] = block
    elif change == "keypoints":
        del group["left.obs_hand_keypoints"]
    elif change == "order":
        features = dict(group.attrs["features"])
        features["left.cmd_hand_joints"]["joint_names"].reverse()
        group.attrs["features"] = features
    elif change == "arm":
        del group["right.cmd_joints"]
    elif change == "nan_joints":
        group["left.cmd_hand_joints"][5, 0] = np.nan
    elif change == "nan_keypoints":
        group["left.obs_hand_keypoints"][5, 0] = np.nan
    report = validate_episode(complete_delivery, ego_overlay=True)
    assert not report.ok, report.text()


def test_explicit_fixed_camera_is_valid_and_identity_is_only_a_warning(complete_delivery):
    group = zarr.open_group(complete_delivery, mode="a")
    del group["obs_head_pose"]
    block = dict(group.attrs["calibration"])
    block["cameras"]["front_1"]["ref_T_cam"] = np.eye(4).tolist()
    group.attrs["calibration"] = block
    report = validate_episode(complete_delivery, ego_overlay=True)
    assert report.ok, report.text()


def test_structural_sample_cli_is_not_a_complete_delivery_pass(complete_delivery, capsys):
    zarr.open_group(complete_delivery, mode="a").attrs["data_status"] = "structural_sample"
    assert main([str(complete_delivery), "--data-status", "--ego-overlay"]) == 3
    assert "status eligible=no" in capsys.readouterr().out


def test_unused_robot_slots_are_ignored_but_joint_limits_are_enforced(complete_delivery, hand_spec, monkeypatch):
    from dataclasses import replace

    from egomimic.rldb.embodiment.registry import KeypointSpec, load_end_effectors

    spec = replace(hand_spec, urdf=None, keypoints=KeypointSpec("mano21", (0, 1)),
                   joint_limits=tuple((-1.0, 1.0) for _ in hand_spec.joint_names))
    entries = load_end_effectors() | {spec.name: spec}
    monkeypatch.setattr("egomimic.rldb.embodiment.embodiment.load_end_effectors", lambda: entries)
    group = zarr.open_group(complete_delivery, mode="a")
    group["left.obs_hand_keypoints"][:, 6:] = np.nan
    assert validate_episode(complete_delivery, ego_overlay=True).ok
    group["left.cmd_hand_joints"][5, 0] = 2
    report = validate_episode(complete_delivery)
    assert any("registry limits" in f.message for f in report.errors)
