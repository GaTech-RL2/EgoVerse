"""Local compatibility correction is immutable, idempotent, and fail-closed."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr


@pytest.fixture(scope="module")
def modules():
    import sys

    folder = (
        Path(__file__).resolve().parents[2] / "experiments/fold_clothes_moving_opscale"
    )
    sys.path.insert(0, str(folder))
    return importlib.import_module("canonicalize"), importlib.import_module(
        "check_readiness"
    )


@pytest.fixture
def raw_episode(tmp_path):
    from egomimic.rldb.zarr.action_chunk_transforms import RotateLocalFrame
    from egomimic.scripts.mecka_process.mecka_to_zarr import compute_hand_pose_xyzquat

    def make(convention="pre_fix"):
        root = tmp_path / "zarr" / f"{convention}.zarr"
        root.parent.mkdir(exist_ok=True)
        group = zarr.open_group(str(root), mode="w", zarr_format=3)
        group.attrs.update(
            {"episode_id": convention, "total_frames": 60, "fps": 30, "duration": 2}
        )
        kp = np.tile([0.0, 0.0, 1.0], (21, 1))
        kp[5], kp[9], kp[17] = [-0.04, 0.02, 1], [0, 0.04, 1], [0.04, 0.02, 1]
        _, pose = compute_hand_pose_xyzquat(kp, 0)
        poses = np.tile(pose, (60, 1))
        left = poses.copy()
        if convention == "pre_fix":
            left = RotateLocalFrame(keys=["left"]).transform({"left": left})["left"]
        elif convention == "unknown":
            left[:, 3:] = [1, 0, 0, 0]
        for side in ("left", "right"):
            group.create_array(
                f"{side}.obs_keypoints", data=np.tile(kp.reshape(1, 63), (60, 1))
            )
            group.create_array(
                f"{side}.obs_wrist_pose", data=left if side == "left" else poses
            )
        return root

    return make


def test_pre_fix_corrected_once_without_touching_raw(modules, raw_episode, tmp_path):
    canonical, _ = modules
    raw = raw_episode("pre_fix")
    before = canonical.tree_digest(raw)
    view = tmp_path / "canonical_zarr"
    result = canonical.canonicalize_episode(raw, view, 60)
    assert result["source_convention"] == "pre_fix"
    assert result["raw_left_sha256"] != result["output_left_sha256"]
    assert (
        canonical.classify(zarr.open_group(str(view / raw.name), mode="r"), 60)
        == "post_fix"
    )
    assert (view / raw.name / "right.obs_wrist_pose").is_symlink()
    assert not (view / raw.name / "left.obs_wrist_pose").is_symlink()
    first = canonical.tree_digest(view / raw.name)
    assert canonical.canonicalize_episode(raw, view, 60) == result
    assert canonical.tree_digest(view / raw.name) == first
    assert canonical.tree_digest(raw) == before


def test_post_fix_array_is_unchanged_link(modules, raw_episode, tmp_path):
    canonical, _ = modules
    raw = raw_episode("post_fix")
    result = canonical.canonicalize_episode(raw, tmp_path / "canonical_zarr", 60)
    assert result["operation"] == "unchanged"
    assert result["raw_left_sha256"] == result["output_left_sha256"]
    assert (tmp_path / "canonical_zarr" / raw.name / "left.obs_wrist_pose").is_symlink()


def test_unknown_convention_is_rejected(modules, raw_episode, tmp_path):
    canonical, _ = modules
    with pytest.raises(ValueError, match="Unknown raw wrist convention"):
        canonical.canonicalize_episode(
            raw_episode("unknown"), tmp_path / "canonical_zarr", 60
        )


def test_modified_canonical_array_is_rejected_on_retry(modules, raw_episode, tmp_path):
    canonical, _ = modules
    raw = raw_episode("pre_fix")
    view = tmp_path / "canonical_zarr"
    canonical.canonicalize_episode(raw, view, 60)
    array = zarr.open_array(str(view / raw.name / "left.obs_wrist_pose"), mode="r+")
    pose = array[0]
    pose[0] += 1
    array[0] = pose
    with pytest.raises(ValueError, match="Canonical left wrist hash differs"):
        canonical.canonicalize_episode(raw, view, 60)


def test_partial_receipt_recovered_without_reapplying(modules, raw_episode, tmp_path):
    canonical, _ = modules
    raw = raw_episode("pre_fix")
    view = tmp_path / "canonical_zarr"
    first = canonical.canonicalize_episode(raw, view, 60)
    (tmp_path / "canonical_provenance" / (raw.stem + ".json")).unlink()
    assert canonical.canonicalize_episode(raw, view, 60) == first


def test_timing_uses_exact_retained_frames_and_still_checks_duration(modules):
    _, readiness = modules
    expected = {"duration": 105.0, "converted_frames": 3145}
    assert (
        readiness.check_timing({"duration": 105.0, "total_frames": 3145}, expected)
        == 3145
    )
    with pytest.raises(ValueError, match="authoritative converted frame index"):
        readiness.check_timing({"duration": 105.0, "total_frames": 3144}, expected)
    with pytest.raises(ValueError, match="manifest disagree"):
        readiness.check_timing({"duration": 104.0, "total_frames": 3145}, expected)


def test_missing_frame_metadata_fails_closed(modules, tmp_path):
    _, readiness = modules
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/main_episodes.csv").write_text(
        "episode_id,user_id,task_id,duration_s,frames\n"
    )
    with pytest.raises(
        ValueError, match="Missing authoritative converted frame counts"
    ):
        readiness.converted_frame_index(tmp_path, {"episodes": {"a": {}}}, {"a"})


def test_zarr_variable_length_annotation_bytes(tmp_path):
    from zarr.core.dtype import VariableLengthBytes

    group = zarr.open_group(str(tmp_path / "bytes.zarr"), mode="w", zarr_format=3)
    array = group.create_array("annotations", shape=(1,), dtype=VariableLengthBytes())
    array[:] = np.array([b'{"text":"fold shirt","start":0,"end":60}'], dtype=object)
    assert json.loads(bytes(array[0:1][0]))["text"] == "fold shirt"
