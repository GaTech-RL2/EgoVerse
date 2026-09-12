"""Legacy omissions are limitations; broken declared representations are errors."""

import numpy as np
import pytest
import zarr

from egomimic.rldb.embodiment import Embodiment
from egomimic.rldb.zarr.test_validate import _poses, _write_eva
from egomimic.rldb.zarr.validate import ERROR, WARNING, validate_episode


@pytest.mark.parametrize(
    "embodiment", ["eva_bimanual", "human_bimanual", "yam_bimanual"]
)
def test_old_episodes_remain_usable_without_rewrite(tmp_path, embodiment):
    path = tmp_path / "old.zarr"
    _write_eva(path)
    group = zarr.open_group(path, mode="a")
    for name in (
        "calibration",
        "intrinsics",
        "extrinsics",
        "schema_version",
        "data_status",
        "morphology",
    ):
        group.attrs.pop(name, None)
    group.attrs["embodiment"] = embodiment
    del group["obs_rgb_timestamps_ns"]
    for side in ("left", "right"):
        for key in ("obs_joints", "cmd_joints"):
            del group[f"{side}.{key}"]
        if embodiment == "human_bimanual":
            group.create_array(f"{side}.obs_keypoints", data=np.zeros((4, 63)))
    before = (path / "zarr.json").read_bytes()
    report = validate_episode(path)
    assert report.ok, report.text()
    assert any(
        f.check == "calibration_present" and f.level == WARNING for f in report.findings
    )
    assert (path / "zarr.json").read_bytes() == before
    assert Embodiment.from_attrs(group.attrs).platform.name in (
        "eva_x5",
        "human_body",
        "yam_x6",
    )
    strict_calibration = validate_episode(
        path, requirements={"calibration_present": True}
    )
    assert any(
        f.check == "calibration_present" and f.level == ERROR
        for f in strict_calibration.findings
    )
    group.create_array("obs_head_pose", data=_poses()[:, :6])
    assert any(
        f.check == "obs_head_pose" and f.level == ERROR
        for f in validate_episode(path).findings
    )


@pytest.mark.parametrize("keypoints", [None, "obs_keypoints", "obs_aria_keypoints"])
@pytest.mark.parametrize("matrix", [True, False])
def test_legacy_human_pose_only_and_aria_with_old_intrinsics(tmp_path, keypoints, matrix):
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter

    path = tmp_path / "human.zarr"
    numeric = {f"{side}.obs_ee_pose": _poses() for side in ("left", "right")}
    if keypoints:
        numeric.update({f"{side}.{keypoints}": np.zeros((4, 63)) for side in ("left", "right")})
    ZarrWriter.create_and_write(path, numeric_data=numeric, embodiment="human_bimanual", chunk_timesteps=4,
                               intrinsics={"front_1": np.eye(3, 4)})
    group = zarr.open_group(path, mode="a")
    K = [[200., 0, 160], [0, 200, 120], [0, 0, 1]]
    group.attrs["intrinsics"] = K if matrix else {"front_1": K}
    before = (path / "zarr.json").read_bytes()
    assert validate_episode(path).ok, validate_episode(path).text()
    sample = ZarrDataset(path, {"pose": {"zarr_key": "left.obs_ee_pose"}})[0]
    np.testing.assert_allclose(sample["pose"], _poses()[0])
    assert (path / "zarr.json").read_bytes() == before
    if keypoints:
        group[f"left.{keypoints}"].resize((4, 62))
        assert not validate_episode(path).ok
