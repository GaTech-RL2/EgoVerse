"""The Mecka converter writes an episode that passes `egoverse validate` (Phase 3).

Everything but the download and the video decode runs for real: episodes are
built from small hands.csv / egomotion.txt / frames.csv / annotations.csv files
via --local-data-dir, with `_extract_video_frames` stubbed."""

import json

import numpy as np
import pandas as pd
import pytest
import zarr

from egomimic.rldb.embodiment.embodiment import EMBODIMENT
from egomimic.rldb.zarr.schema import validate_episode
from egomimic.scripts.mecka_process import mecka_to_zarr as m


def test_arm_flag_maps_to_current_human_embodiments() -> None:
    assert set(m.ARM_TO_EMBODIMENT) == {"both", "left", "right"}
    for name in m.ARM_TO_EMBODIMENT.values():
        assert name.startswith("human_") and name.upper() in EMBODIMENT.__members__
    with pytest.raises(ValueError, match="arm must be one of"):
        m.MeckaDatasetConverter("unused.json", "out", "mecka/demo", arm="mecka")


def _read_annotations(group) -> list[dict]:
    """ZarrWriter._write_annotations stores one JSON object per row as bytes
    under the `annotations` key: {"text", "start_idx", "end_idx"}."""
    out = []
    for raw in group["annotations"][:]:
        text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8")
        out.append(json.loads(text))
    return out


def _convert(tmp_path, monkeypatch, **kwargs):
    """Run MeckaDatasetConverter on a raw episode whose video is one frame
    shorter than frames.csv / egomotion.txt (so the extractor syncs every
    stream, head pose included, to 4 frames), with the right hand undetected in
    frame 2, one annotation running past the end and one with a blank label."""
    n_video, n_rows = 4, 5
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "episode.json").write_text(json.dumps({"id": "ep-1", "urls": {}}))
    (raw / "video.mp4").touch()
    pd.DataFrame({"frame": range(n_rows)}).to_csv(raw / "frames.csv", index=False)
    quats = rng.normal(size=(n_rows, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)  # xyzw
    egomotion = np.zeros((n_rows, 11))
    egomotion[:, 0] = np.arange(n_rows)
    egomotion[:, 1:4] = rng.normal(size=(n_rows, 3))
    egomotion[:, 7:11] = quats
    np.savetxt(raw / "egomotion.txt", egomotion)
    hands = [
        (f, h, lm, *(rng.normal(size=3) * 0.05 + [0, 0, 0.5]))
        for f in range(n_rows)
        for h in (0, 1)
        if (f, h) != (2, 1)
        for lm in range(21)
    ]
    pd.DataFrame(
        hands,
        columns=[
            "frame",
            "hand_index",
            "landmark_index",
            "world_x",
            "world_y",
            "world_z",
        ],
    ).to_csv(raw / "hands.csv", index=False)
    pd.DataFrame(
        [
            {"label": "fold_towel", "start_time": 0.0, "end_time": 0.2},
            {"label": None, "start_time": 0.0, "end_time": 0.1},
        ]
    ).to_csv(raw / "annotations.csv", index=False)
    frames = rng.integers(0, 255, size=(n_video, 36, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(
        m.MeckaExtractor, "_extract_video_frames", staticmethod(lambda p, n: frames)
    )
    conv = m.MeckaDatasetConverter(
        str(raw / "episode.json"),
        str(tmp_path / "out"),
        "mecka/demo",
        local_data_dir=raw,
        save_mp4=False,
        **kwargs,
    )
    return conv.extract_episode(), raw


def test_converted_episode_passes_validation(tmp_path, monkeypatch) -> None:
    # Needs the validator to skip all-zero pose rows as missing frames
    # (ryanco/phase2-format-version).
    path, _ = _convert(tmp_path, monkeypatch, task_name="fold_clothes")
    rep = validate_episode(path)
    assert not rep.errors, rep.errors


def test_converted_episode_contents(tmp_path, monkeypatch) -> None:
    path, raw = _convert(tmp_path, monkeypatch, task_description="fold a towel")
    assert path == tmp_path / "out" / "ep-1.zarr"
    g = zarr.open_group(str(path), mode="r")
    attrs = dict(g.attrs)
    assert attrs["embodiment"] == "human_bimanual"
    assert attrs["task_name"] == ""
    assert attrs["task_description"] == "fold a towel"
    assert attrs["total_frames"] == 4
    np.testing.assert_allclose(attrs["intrinsics"]["front_1"], m.MECKA_INTRINSICS)
    assert attrs["provenance"]["converter"] == m.__name__
    assert attrs["provenance"]["source_uri"] == raw.resolve().as_uri()
    for key in ("right.obs_ee_pose", "right.obs_wrist_pose"):
        arr = np.asarray(g[key][:4])
        assert not arr[2].any()  # undetected hand -> all-zero row
        np.testing.assert_allclose(
            np.linalg.norm(np.delete(arr, 2, axis=0)[:, 3:], axis=1), 1.0
        )
    # 0.2 s * 30 fps = frame 6, clipped to the 4-frame episode; blank label dropped.
    rows = _read_annotations(g)
    assert [(r["text"], r["start_idx"], r["end_idx"]) for r in rows] == [
        ("fold towel", 0, 4)
    ]


def test_annotation_rows_become_frame_spans(tmp_path) -> None:
    T = 40
    pose = np.tile([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], (T, 1))
    feats = {
        "left.obs_ee_pose": pose,
        "right.obs_ee_pose": pose,
        "images.front_1": np.zeros((T, 16, 16, 3), np.uint8),
    }
    ann = pd.DataFrame(
        [
            {"label": "pick_up_cup", "start_time": 0.5, "end_time": 1.0},
            {"label": "put_down_cup", "start_time": 1.2, "end_time": 2.0},
            {"label": "leave", "start_time": 1.5, "end_time": 2.0},
        ]
    )
    path = m.write_episode_zarr(
        feats, ann, "ep-2", tmp_path, embodiment="human_bimanual", task_name="t"
    )
    rows = _read_annotations(zarr.open_group(str(path), mode="r"))
    assert [(r["text"], r["start_idx"], r["end_idx"]) for r in rows] == [
        ("pick up cup", 15, 30),
        ("put down cup", 36, 40),
    ]
