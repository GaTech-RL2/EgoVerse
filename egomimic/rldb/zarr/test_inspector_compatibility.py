"""Inspector/artifact parity, real HTTP routes, and degraded overlay imports."""

import json
import subprocess
import sys

import numpy as np
import pytest
import zarr

from egomimic.rldb.zarr.overlay import (
    decode_frame,
    keypoint_chunk,
    pose_chunk,
    render_overlay,
)
from egomimic.rldb.zarr.render_check import main, render_episode
from egomimic.rldb.zarr.test_render_check import overlay_episode as overlay_episode
from egomimic.scripts.data_visualization.inspector_lib import dataset_view


@pytest.mark.parametrize("mode", ["cartesian", "orientation", "keypoint"])
def test_inspector_and_artifact_share_same_pixels_and_tail(overlay_episode, tmp_path, mode):
    group = zarr.open_group(overlay_episode, mode="a")
    for side in ("left", "right"):
        group[f"{side}.obs_ee_pose"][:, 2] = 1
    for frame in (0, 1, 3):
        image = decode_frame(group, frame)
        expected, _ = render_overlay(group, frame, image=image, mode=mode, horizon=3)
        actual, ok, _ = dataset_view._draw_overlay(image, group, frame, mode, horizon=3)
        assert ok
        np.testing.assert_array_equal(actual, expected)
    report = render_episode(overlay_episode, tmp_path / f"{mode}.mp4", mode=mode, horizon=3)
    assert report["overlay_frames"] == 4 and report["mode"] == mode


def test_pose_only_artifact_and_explicit_missing_keypoints(overlay_episode, tmp_path, capsys):
    group = zarr.open_group(overlay_episode, mode="a")
    for side in ("left", "right"):
        del group[f"{side}.obs_keypoints"]
        group[f"{side}.obs_ee_pose"][:, 2] = 1
    out = tmp_path / "poses.mp4"
    assert main([str(overlay_episode), "--mode", "cartesian", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Validation: 0 errors" in text and "data_status=complete" in text
    assert main([str(overlay_episode), "--mode", "keypoint", "--out", str(out)]) == 2
    assert "supplied keypoints" in capsys.readouterr().out


def test_raw_aria_alias_uses_aria_topology_without_reordering(overlay_episode, monkeypatch):
    from egomimic.rldb.embodiment.embodiment import ResolvedEmbodiment
    from egomimic.rldb.embodiment.human import ARIA_FINGER_EDGES

    group = zarr.open_group(overlay_episode, mode="a")
    points = np.asarray(group["left.obs_keypoints"][:])
    group.create_array("left.obs_aria_keypoints", data=points)
    del group["left.obs_keypoints"]
    np.testing.assert_allclose(keypoint_chunk(group, 0)[2][0, :63], points[0])
    calls = []
    original = ResolvedEmbodiment.viz

    def capture(self, image, values, **kwargs):
        calls.append(kwargs)
        return original(self, image, values, **kwargs)

    monkeypatch.setattr(ResolvedEmbodiment, "viz", capture)
    _, diagnostic = render_overlay(group, 0)
    assert calls[0]["finger_edges"] == ARIA_FINGER_EDGES
    assert calls[0]["label_slot"] == 5
    assert diagnostic["keypoint_sources"]["left"] == "left.obs_aria_keypoints"
    figure = dataset_view.build_3d_figure(group, 0, "keypoint")
    skeleton = next(t for t in figure.data if t.name == "left skeleton")
    assert len(skeleton.x) == 3 * len(ARIA_FINGER_EDGES)


def test_moving_camera_pose_horizon_matches_keypoint_geometry(overlay_episode):
    group = zarr.open_group(overlay_episode, mode="a")
    group["obs_head_pose"][:, 0] = np.arange(6) * 0.1
    for side in ("left", "right"):
        group[f"{side}.obs_ee_pose"][:, 2] = 1
    before = pose_chunk(group, 1, horizon=9)[2]
    assert len(before) == 3
    np.testing.assert_allclose(before[:, 0], -0.1)
    group["obs_head_pose"][2:, 0] = 20
    np.testing.assert_allclose(before, pose_chunk(group, 1, horizon=9)[2])


def test_http_routes_and_cache_include_image_camera_and_horizon(overlay_episode):
    from egomimic.scripts.data_visualization.inspector_lib.app import build_dataset_app

    app = build_dataset_app(str(overlay_episode.parent))
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    layout = client.get("/_dash-layout")
    assert b"ds_horizon" in layout.data and b"ds_play" in layout.data
    for mode in ("none", "keypoint", "cartesian", "orientation"):
        response = client.get(f"/dataset_frame/{overlay_episode.name}/1?overlay={mode}&horizon=3&annot=1")
        assert response.status_code == 200 and response.mimetype == "image/jpeg"
    assert client.get(f"/dataset_frame/{overlay_episode.name}/1?horizon=bad").status_code == 400
    assert any(key[-1] == 3 for key in dataset_view._RENDER_CACHE)


def test_missing_embodiment_dependency_keeps_rgb_available_in_fresh_process(overlay_episode):
    script = '''
import builtins, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith("egomimic.rldb.embodiment"):
        raise ImportError("intentional reduced dependency fixture")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from pathlib import Path
from egomimic.scripts.data_visualization.inspector_lib import dataset_view as view
path = Path(sys.argv[1])
assert view.Embodiment is None
image = view.render_frame_jpeg(str(path.parent), path.name, 0, overlay="keypoint", annotate=True, image_key="images.front_1")
assert image.startswith(bytes([255, 216]))
print(json.dumps({"rgb_available": True}))
'''
    result = subprocess.run([sys.executable, "-c", script, str(overlay_episode)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["rgb_available"]


def test_inspector_cache_does_not_reuse_pixels_from_another_camera(overlay_episode):
    import simplejpeg
    from zarr.core.dtype import VariableLengthBytes

    group = zarr.open_group(overlay_episode, mode="a")
    jpeg = simplejpeg.encode_jpeg(np.full((64, 64, 3), 180, np.uint8), colorspace="RGB")
    group.create_array("images.left_wrist", shape=(4,), dtype=VariableLengthBytes())
    group["images.left_wrist"][:] = np.array([jpeg] * 4, dtype=object)
    kw = dict(overlay="none", annotate=False)
    front = dataset_view.render_frame_jpeg(str(overlay_episode.parent), overlay_episode.name, 0,
                                           image_key="images.front_1", **kw)
    wrist = dataset_view.render_frame_jpeg(str(overlay_episode.parent), overlay_episode.name, 0,
                                           image_key="images.left_wrist", **kw)
    assert front != wrist


def test_3d_missing_point_preserves_slots_and_incident_edges(overlay_episode):
    group = zarr.open_group(overlay_episode, mode="a")
    group["left.obs_keypoints"][0, 3:6] = np.nan
    figure = dataset_view.build_3d_figure(group, 0, "keypoint")
    points = next(t for t in figure.data if t.name == "left keypoints")
    skeleton = next(t for t in figure.data if t.name == "left skeleton")
    assert len(points.x) == 21 and np.isnan(points.x[1])
    assert len(skeleton.x) == (20 - 2) * 3
    assert points.marker.color == "rgb(255,120,0)"
