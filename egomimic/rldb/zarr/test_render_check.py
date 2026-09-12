"""Exercise actual Zarr/JPEG/MP4 I/O and the shared keypoint overlay."""

import av
import numpy as np
import pytest
import zarr

from egomimic.rldb.zarr.overlay import (
    OverlayUnavailable,
    decode_frame,
    render_keypoints,
)
from egomimic.rldb.zarr.render_check import main, render_episode
from egomimic.rldb.zarr.zarr_writer import ZarrWriter


@pytest.fixture
def overlay_episode(tmp_path):
    path = tmp_path / "human.zarr"
    pose = np.tile([0, 0, 0, 1, 0, 0, 0], (4, 1)).astype(float)
    numeric = {"obs_head_pose": pose}
    for side in ("left", "right"):
        points = np.zeros((4, 21, 3))
        points[:, :, 2] = 1
        points[:, :, 0] = np.linspace(-0.3, 0.3, 21)
        numeric[f"{side}.obs_keypoints"] = points.reshape(4, 63)
        numeric[f"{side}.obs_ee_pose"] = pose.copy()
    ZarrWriter.create_and_write(
        path,
        numeric_data=numeric,
        image_data={"images.front_1": np.zeros((4, 64, 64, 3), np.uint8)},
        embodiment="human_bimanual",
        chunk_timesteps=3,
        intrinsics={
            "front_1": np.array([[50, 0, 32, 0], [0, 50, 32, 0], [0, 0, 1, 0]])
        },
    )
    return path


def test_preview_stops_at_total_frames_and_writes_report(overlay_episode, tmp_path):
    out = tmp_path / "preview.mp4"
    report = render_episode(overlay_episode, out, horizon=30)
    assert len(report["frames"]) == report["overlay_frames"] == 4
    assert report["frames"][0]["points"] == 4 * 42
    assert report["frames"][-1]["points"] == 42
    with av.open(str(out)) as video:
        images = list(video.decode(video=0))
    assert len(images) == 4
    assert images[0].to_ndarray(format="rgb24").any()
    assert out.with_suffix(".json").is_file()


def test_unavailable_overlay_has_video_report_and_cli_status(overlay_episode, tmp_path):
    group = zarr.open_group(overlay_episode, mode="a")
    group.attrs.pop("intrinsics")
    out = tmp_path / "missing.mp4"
    assert main([str(overlay_episode), "--out", str(out), "--max-frames", "1"]) == 2
    import json

    report = json.loads(out.with_suffix(".json").read_text())
    assert "intrinsics" in report["frames"][0]["reason"]
    assert out.is_file()


def test_robot_hand_keypoints_are_loaded_without_human_arrays(overlay_episode):
    group = zarr.open_group(overlay_episode, mode="a")
    group.attrs["embodiment"] = "dexmate_bimanual"
    for side in ("left", "right"):
        group.create_array(
            f"{side}.obs_hand_keypoints",
            data=np.asarray(group[f"{side}.obs_keypoints"][:]),
        )
        del group[f"{side}.obs_keypoints"]
    image, diagnostic = render_keypoints(group, 1, horizon=2)
    assert image.any()
    assert diagnostic["inside_image"] == 84
    with pytest.raises(OverlayUnavailable):
        decode_frame(group, 4)
