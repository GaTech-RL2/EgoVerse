"""Check batched Sharpa forward kinematics against the per-frame derivation."""

import numpy as np
import pytest
import zarr

from egomimic.rldb.embodiment import Embodiment
from egomimic.rldb.embodiment.hand_kinematics import fk_keypoints, load_chain
from egomimic.rldb.embodiment.urdf import load_urdf
from egomimic.rldb.zarr.sharpa_preview import reconstruct_tracks
from egomimic.scripts.sharpa_process.sharpa_tracks import (
    SharpaKinematics,
    clip_to_limits,
    derive_tracks,
    joint_limit_excursions,
)
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz

FRAMES = 5


@pytest.fixture(scope="module")
def kinematics():
    return SharpaKinematics()


def _in_limit_vector(kinematics, frames, seed):
    """Return deterministic joint vectors strictly inside every limit."""
    rng = np.random.default_rng(seed)
    lower, upper = kinematics.limits.T
    return lower + (upper - lower) * rng.uniform(0.1, 0.9, size=(frames, 65))


def _per_frame_reference(observed, commanded):
    """The original one-configuration-at-a-time derivation in sharpa_preview."""
    resolved = Embodiment.resolve("dexmate_bimanual")
    platform = load_urdf(resolved.platform.asset_path(verify=True))
    total = len(observed)
    output = {}
    observed_head = None
    for prefix, raw in (("obs", observed), ("cmd", commanded)):
        output[f"{prefix}_aux_joints"] = raw[:, 58:65]
        values = {
            name: raw[:, 58 + i]
            for i, name in enumerate(resolved.platform.aux.joint_names)
        }
        for side, offset in (("left", 0), ("right", 29)):
            for j in range(7):
                values[f"{side}_arm_joint_{j + 1}"] = raw[:, offset + j]
            output[f"{side}.{prefix}_joints"] = raw[:, offset : offset + 7]
            output[f"{side}.{prefix}_hand_joints"] = raw[:, offset + 7 : offset + 29]
        poses = [
            platform.link_transforms({k: v[t] for k, v in values.items()})
            for t in range(total)
        ]
        if prefix == "obs":
            observed_head = np.stack([p["head_base_link"] for p in poses])
        for side in ("left", "right"):
            spec = resolved.end_effectors[side]
            chain = load_chain(spec)
            joints = output[f"{side}.{prefix}_hand_joints"]
            palm = np.stack(
                [
                    p[f"{side}_hand_base_link"]
                    @ chain.link_transforms(dict(zip(spec.joint_names, row)))[
                        spec.ee_pose_link
                    ]
                    for p, row in zip(poses, joints, strict=True)
                ]
            )
            output[f"{side}.{prefix}_ee_pose"] = _matrix_to_xyzwxyz(palm)
            if prefix == "obs":
                local = fk_keypoints(spec, joints)
                points = (
                    np.einsum("tij,tkj->tki", palm[:, :3, :3], local)
                    + palm[:, None, :3, 3]
                )
                output[f"{side}.obs_hand_keypoints"] = points.reshape(total, 63)
    return output, observed_head


def test_batched_tracks_match_the_per_frame_derivation(kinematics, tmp_path):
    observed = _in_limit_vector(kinematics, FRAMES, 0)
    commanded = _in_limit_vector(kinematics, FRAMES, 1)
    expected, expected_head = _per_frame_reference(observed, commanded)

    group = zarr.open_group(tmp_path / "sample.zarr", mode="w")
    group.create_array("sharpa.observation_state", data=observed)
    group.create_array("sharpa.action", data=commanded)
    group.attrs["total_frames"] = FRAMES
    tracks, head = reconstruct_tracks(group)

    assert set(tracks) == set(expected)
    for key, values in expected.items():
        assert tracks[key].dtype == np.float64, key
        np.testing.assert_allclose(tracks[key], values, rtol=0, atol=1e-9, err_msg=key)
    assert head.shape == (FRAMES, 4, 4)
    np.testing.assert_allclose(head, expected_head, rtol=0, atol=1e-9)


def test_batches_and_hand_links_compose_consistently(kinematics, monkeypatch):
    observed = _in_limit_vector(kinematics, 7, 2)
    full, links = derive_tracks(
        observed, observed, kinematics=kinematics, links=("left_hand_wrist",)
    )
    monkeypatch.setattr("egomimic.scripts.sharpa_process.sharpa_tracks.FK_BLOCK", 3)
    blocked, blocked_links = derive_tracks(
        observed, observed, kinematics=kinematics, links=("left_hand_wrist",)
    )
    for key, values in full.items():
        np.testing.assert_array_equal(blocked[key], values, err_msg=key)
    np.testing.assert_array_equal(
        blocked_links["left_hand_wrist"], links["left_hand_wrist"]
    )
    # Observed and commanded tracks agree when the vectors agree.
    np.testing.assert_array_equal(full["left.obs_ee_pose"], full["left.cmd_ee_pose"])
    assert not kinematics.is_fixed_in_base("head_base_link")
    assert kinematics.is_fixed_in_base("lower_body_base_link")


def test_excursions_are_reported_and_clipped_per_column(kinematics):
    raw = _in_limit_vector(kinematics, 4, 3).astype(np.float32)
    column = kinematics.column_names.index("right_middle_DIP")
    upper = kinematics.limits[column, 1]
    raw[1:3, column] = upper + 0.15
    records = joint_limit_excursions(raw, "obs", kinematics)
    assert len(records) == 1
    record = records[0]
    assert record["array"] == "right.obs_hand_joints"
    assert record["frames_above"] == 2 and record["frames_below"] == 0
    assert record["max_above"] == pytest.approx(0.15, abs=1e-6)
    assert record["validator_checked"]
    clipped = clip_to_limits(raw, kinematics)
    assert clipped.dtype == np.float32
    assert joint_limit_excursions(clipped, "obs", kinematics) == []
    assert np.array_equal(np.delete(clipped, column, 1), np.delete(raw, column, 1))
