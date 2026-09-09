"""YAM embodiment, ABC image/intrinsics harmonisation, and the ABC graph config."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_io import ActionTargetBuilder
from egomimic.rldb.embodiment.embodiment import EMBODIMENT, get_embodiment, get_embodiment_id
from egomimic.rldb.embodiment.yam import Yam
from egomimic.rldb.zarr.action_chunk_transforms import (
    BatchQuaternionPoseTo6D,
    QuaternionPoseTo6D,
    XYZWXYZ_to_XYZRot6D,
    XYZWXYZ_to_XYZYPR,
)
from egomimic.rldb.zarr.zarr_dataset_multi import (
    _image_hw_of,
    _resize_images,
    _scale_intrinsics,
)

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "config_graph_abc", _REPO / "tools/config_graph.py"
)
config_graph = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(config_graph)


# -- embodiment registration ------------------------------------------------


def test_yam_bimanual_is_registered_and_round_trips():
    assert get_embodiment_id("yam_bimanual") == EMBODIMENT.YAM_BIMANUAL.value
    assert get_embodiment(EMBODIMENT.YAM_BIMANUAL.value) == "YAM_BIMANUAL"


def test_yam_carries_no_class_intrinsics():
    # Two station types with different calibration ship in ABC-130k, so K is
    # per-episode; a class constant would silently apply one station's K to both.
    assert Yam.INTRINSICS is None


# -- transform axes ---------------------------------------------------------


@pytest.mark.parametrize("coord_frame", ["camframe", "world", "eef_frame"])
@pytest.mark.parametrize("rotation_mode", ["euler", "quat", "6D"])
def test_yam_builds_every_frame_and_rotation_combination(coord_frame, rotation_mode):
    transforms = Yam.get_transform_list(
        action_mode="cartesian", coord_frame=coord_frame, rotation_mode=rotation_mode
    )
    assert transforms
    kinds = {type(t) for t in transforms}
    assert (XYZWXYZ_to_XYZRot6D in kinds) == (rotation_mode == "6D")
    assert (XYZWXYZ_to_XYZYPR in kinds) == (rotation_mode == "euler")


def test_yam_rejects_unknown_axes():
    with pytest.raises(ValueError, match="unknown coord_frame"):
        Yam.get_transform_list(coord_frame="headframe")
    with pytest.raises(ValueError, match="unknown action_mode"):
        Yam.get_transform_list(action_mode="keypoints")


def test_yam_keymap_covers_all_three_cameras_and_both_arms():
    keymap = Yam.get_keymap(keymap_mode="cartesian")
    zarr_keys = {spec["zarr_key"] for spec in keymap.values()}
    assert {"images.front_1", "images.right_wrist", "images.left_wrist"} <= zarr_keys
    assert {"right.obs_ee_pose", "left.obs_ee_pose"} <= zarr_keys


# -- 6D pose transforms -----------------------------------------------------


def _pose(n):
    pose = np.zeros((n, 7))
    pose[:, 3] = 1.0  # identity quaternion, wxyz
    pose[:, :3] = np.arange(3 * n).reshape(n, 3)
    return pose


def test_batch_quaternion_pose_to_6d_emits_identity_columns():
    out = BatchQuaternionPoseTo6D(pose_key="p", output_key="q").transform({"p": _pose(2)})
    assert out["q"].shape == (2, 9)
    # Identity rotation -> first two columns of I3.
    np.testing.assert_allclose(out["q"][:, 3:9], np.tile([1, 0, 0, 0, 1, 0], (2, 1)))


def test_single_quaternion_pose_to_6d_emits_nine_values():
    out = QuaternionPoseTo6D(pose_key="p", output_key="q").transform({"p": _pose(1)[0]})
    assert out["q"].shape == (9,)


def test_quaternion_pose_to_6d_rejects_a_batch():
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        QuaternionPoseTo6D(pose_key="p", output_key="q").transform({"p": _pose(3)})


def test_batch_quaternion_pose_to_6d_rejects_a_single_pose():
    with pytest.raises(ValueError, match=r"shape \(N, 7\)"):
        BatchQuaternionPoseTo6D(pose_key="p", output_key="q").transform({"p": _pose(1)[0]})


# -- image harmonisation ----------------------------------------------------


def test_image_hw_reads_channels_first_without_confusing_channels_for_height():
    assert _image_hw_of(np.zeros((3, 480, 640))) == (480, 640)
    assert _image_hw_of(np.zeros((480, 640, 3))) == (480, 640)


def test_image_hw_handles_a_window_in_both_layouts():
    assert _image_hw_of(np.zeros((4, 3, 480, 640))) == (480, 640)
    assert _image_hw_of(np.zeros((4, 480, 640, 3))) == (480, 640)


def test_resize_preserves_layout_and_dtype():
    chw = np.zeros((3, 360, 640), dtype=np.float32)
    assert _resize_images(chw, (480, 640)).shape == (3, 480, 640)
    hwc = np.zeros((360, 640, 3), dtype=np.uint8)
    out = _resize_images(hwc, (480, 640))
    assert out.shape == (480, 640, 3) and out.dtype == np.uint8


def test_resize_is_a_no_op_at_the_target_size():
    arr = np.zeros((3, 480, 640), dtype=np.float32)
    assert _resize_images(arr, (480, 640)) is arr


def test_resize_maps_a_window_frame_by_frame():
    assert _resize_images(np.zeros((2, 3, 360, 640)), (480, 640)).shape == (2, 3, 480, 640)


def test_intrinsics_scale_per_axis_not_by_one_shared_factor():
    K = np.array(
        [[500.0, 0.0, 320.0, 0.0], [0.0, 500.0, 180.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    out = _scale_intrinsics(K, (360, 640), (480, 640))
    # Width is unchanged, height grows 4/3: row 0 must not move, row 1 must.
    np.testing.assert_allclose(out[0], K[0])
    np.testing.assert_allclose(out[1], K[1] * (480 / 360))


def test_intrinsics_scaling_tolerates_a_degenerate_source_size():
    K = np.eye(3, 4, dtype=np.float32)
    np.testing.assert_allclose(_scale_intrinsics(K, (0, 0), (480, 640)), K)


# -- the graph itself -------------------------------------------------------


def test_action_target_builder_reads_the_configured_action_key():
    stage = ActionTargetBuilder(action_key="actions_cartesian")
    assert stage.contract("train") == (("actions_cartesian",), ("target",))
    out = stage.forward({"actions_cartesian": 7})
    assert out["target"] == 7 and "actions_cartesian" not in out


def test_action_target_builder_still_defaults_to_actions():
    assert ActionTargetBuilder().contract("train") == (("actions",), ("target",))


def test_action_target_builder_names_the_key_it_wanted():
    with pytest.raises(ValueError, match="actions_cartesian"):
        ActionTargetBuilder(action_key="actions_cartesian").forward({"actions": 1})


def test_action_target_builder_is_excluded_from_the_inference_graph():
    pipeline = Pipeline([ActionTargetBuilder(action_key="actions_cartesian")])
    runnable, excluded = pipeline.plan(["actions_cartesian"], mode="inference")
    assert not runnable and excluded[0][1] == ["<train-only>"]


@pytest.mark.parametrize("mode", ["train", "inference"])
def test_abc_experiment_graph_lints_clean(mode):
    graph = config_graph.build_graph(
        _REPO / "egomimic/hydra_configs/experiment/abc/yam_fstshirt_dp.yaml", mode=mode
    )
    assert graph["lint"] == []
    assert graph["seed_keys_by_source"]["yam_bimanual"]


def test_abc_train_graph_has_the_five_dp_stages_and_a_loss():
    graph = config_graph.build_graph(
        _REPO / "egomimic/hydra_configs/experiment/abc/yam_fstshirt_dp.yaml", mode="train"
    )
    assert [node["t"] for node in graph["nodes"]] == [
        "FusedObsEncoder",
        "ActionTargetBuilder",
        "DiffusionNoisingStage",
        "DiffusionDenoiserStage",
        "DiffusionEpsilonLossStage",
    ]
    assert any(key.startswith("loss/") for node in graph["nodes"] for key in node["out"])


def test_abc_inference_graph_drops_the_training_only_stages():
    graph = config_graph.build_graph(
        _REPO / "egomimic/hydra_configs/experiment/abc/yam_fstshirt_dp.yaml",
        mode="inference",
    )
    skipped = {entry["t"] for entry in graph["skipped_stages"]}
    assert skipped == {
        "ActionTargetBuilder",
        "DiffusionNoisingStage",
        "DiffusionEpsilonLossStage",
    }
    assert "pred_action" in {key for node in graph["nodes"] for key in node["out"]}
