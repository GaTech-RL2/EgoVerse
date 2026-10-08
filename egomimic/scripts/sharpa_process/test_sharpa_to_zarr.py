"""Convert tiny synthetic Sharpa LeRobot v2.1/v3.0 seasons end to end."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
import zarr

from egomimic.rldb.embodiment.urdf import load_urdf
from egomimic.rldb.zarr import validate
from egomimic.scripts.sharpa_process import sharpa_to_zarr
from egomimic.scripts.sharpa_process.lerobot_reader import (
    load_episode,
    open_season,
)
from egomimic.scripts.sharpa_process.sharpa_to_zarr import (
    ConversionError,
    load_calibration_file,
)
from egomimic.scripts.sharpa_process.sharpa_tracks import SharpaKinematics
from egomimic.utils.aws.aws_sql import (
    episode_hash_to_timestamp_ms,
    timestamp_ms_to_episode_hash,
)
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix

FPS = 30
SIZE = 96
LENGTHS = (24, 20)
RIG = "POC99001"
SEASON = f"season_{RIG}_2026_01_02_03_04_05_train"
#: 2026-01-02 03:04:05 at +08:00, as UTC nanoseconds.
SEASON_START_NS = (
    int(datetime(2026, 1, 1, 19, 4, 5, tzinfo=timezone.utc).timestamp()) * 10**9
)
UTC_COLUMN = "observation.utc_timestamp_ns"
VENDOR_START_NS = 1_767_300_000_123_456_789
CAMERAS = ("head_left", "wrist_left", "wrist_right", "tactile_deform")
LIMITED_JOINT = "left_middle_DIP"
#: Head link x forward, z up -> optical +X right, +Y down, +Z forward.
OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)


@pytest.fixture(scope="module")
def kinematics():
    return SharpaKinematics()


def _mount(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _joints(kinematics, length, phase):
    """Deterministic in-limit 65-column vectors that move every joint."""
    lower, upper = kinematics.limits.T
    t = np.arange(length)[:, None]
    wave = np.sin(0.2 * t + np.arange(65)[None] + phase)
    return ((lower + upper) / 2 + 0.25 * (upper - lower) * wave).astype(np.float32)


def _frame(global_index, camera_index):
    image = np.empty((SIZE, SIZE, 3), np.uint8)
    image[...] = (global_index * 7 + camera_index * 50) % 200 + 20
    image[:, : SIZE // 2, 1] = 255 - image[:, : SIZE // 2, 0]
    return image


def _write_video(path: Path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = stream.height = SIZE
        stream.pix_fmt = "yuv420p"
        stream.options = {"qp": "0", "preset": "ultrafast"}
        for index, image in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _list_column(values):
    values = np.asarray(values)
    offsets = np.arange(0, values.size + 1, values.shape[1], dtype=np.int32)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(values.ravel()))


def _episode_table(kinematics, episode, length, offset, excursion=None):
    observed = _joints(kinematics, length, episode)
    commanded = _joints(kinematics, length, episode + 0.5)
    if excursion is not None:
        column = kinematics.column_names.index(LIMITED_JOINT)
        observed[2:5, column] = kinematics.limits[column, 1] + excursion
    rng = np.random.default_rng(episode)
    frames = np.arange(length)
    steps = 33_333_333 + (np.arange(offset, offset + length) % 3)
    utc = VENDOR_START_NS + 10**10 * episode + np.cumsum(steps) - steps[0]
    return pa.table(
        {
            "observation.state": _list_column(observed),
            "action": _list_column(commanded),
            "observation.state.joint_torque": _list_column(
                rng.normal(size=(length, 65)).astype(np.float32)
            ),
            "observation.state.tcp": _list_column(
                rng.normal(size=(length, 24)).astype(np.float32)
            ),
            "observation.tactile": _list_column(
                rng.normal(size=(length, 60)).astype(np.float32)
            ),
            UTC_COLUMN: pa.array(utc.astype(np.int64)),
            "timestamp": pa.array((frames / FPS).astype(np.float32)),
            "frame_index": pa.array(frames),
            "episode_index": pa.array(np.full(length, episode)),
            "index": pa.array(offset + frames),
            "task_index": pa.array(np.zeros(length, np.int64)),
        }
    )


def _info(version):
    features = {
        name: {"dtype": "float32", "shape": [width]}
        for name, width in (
            ("observation.state", 65),
            ("action", 65),
            ("observation.state.joint_torque", 65),
            ("observation.state.tcp", 24),
            ("observation.tactile", 60),
        )
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [SIZE, SIZE, 3],
            "info": {"video.codec": "h264", "video.fps": FPS},
        }
    for name in ("frame_index", "episode_index", "index", "task_index"):
        features[name] = {"dtype": "int64", "shape": [1]}
    if version == "v2.1":
        paths = {
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        }
    else:
        paths = {
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
    return {
        "codebase_version": version,
        "robot_type": "north_ces",
        "fps": FPS,
        "chunks_size": 1000,
        "total_episodes": len(LENGTHS),
        **paths,
        "features": features,
    }


def build_dataset(root: Path, kinematics, *, season=SEASON, excursion=None):
    """Write one season in both layouts with identical rows and frames."""
    season_dir = root / season
    offsets = np.concatenate([[0], np.cumsum(LENGTHS)[:-1]]).tolist()
    tables = [
        _episode_table(kinematics, e, n, o, excursion if e == 0 else None)
        for e, (n, o) in enumerate(zip(LENGTHS, offsets))
    ]
    frames = {
        camera: [_frame(g, c) for g in range(sum(LENGTHS))]
        for c, camera in enumerate(CAMERAS)
    }

    v21 = season_dir / "lerobotv2.1"
    (v21 / "meta").mkdir(parents=True)
    (v21 / "meta/info.json").write_text(json.dumps(_info("v2.1")))
    (v21 / "meta/tasks.jsonl").write_text(
        '{"task_index": 0, "task": "north ces task"}\n'
    )
    (v21 / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": e, "tasks": ["north ces task"], "length": n})
            + "\n"
            for e, n in enumerate(LENGTHS)
        )
    )
    for e, (table, n, o) in enumerate(zip(tables, LENGTHS, offsets)):
        data = v21 / f"data/chunk-000/episode_{e:06d}.parquet"
        data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data)
        for camera in CAMERAS:
            _write_video(
                v21
                / f"videos/chunk-000/observation.images.{camera}/episode_{e:06d}.mp4",
                frames[camera][o : o + n],
            )

    v30 = season_dir / "lerobot3.0"
    (v30 / "meta/episodes/chunk-000").mkdir(parents=True)
    (v30 / "meta/info.json").write_text(json.dumps(_info("v3.0")))
    pq.write_table(
        pa.table({"task_index": [0], "task": ["north ces task"]}),
        v30 / "meta/tasks.parquet",
    )
    rows = []
    for e, (n, o) in enumerate(zip(LENGTHS, offsets)):
        row = {
            "episode_index": e,
            "tasks": ["north ces task"],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": o,
            "dataset_to_index": o + n,
        }
        for camera in CAMERAS:
            prefix = f"videos/observation.images.{camera}"
            row.update(
                {
                    f"{prefix}/chunk_index": 0,
                    f"{prefix}/file_index": 0,
                    f"{prefix}/from_timestamp": o / FPS,
                    f"{prefix}/to_timestamp": (o + n) / FPS,
                }
            )
        rows.append(row)
    pq.write_table(
        pa.Table.from_pylist(rows), v30 / "meta/episodes/chunk-000/file-000.parquet"
    )
    data = v30 / "data/chunk-000/file-000.parquet"
    data.parent.mkdir(parents=True)
    pq.write_table(pa.concat_tables(tables), data)
    for camera in CAMERAS:
        _write_video(
            v30 / f"videos/observation.images.{camera}/chunk-000/file-000.mp4",
            frames[camera],
        )
    return root


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, kinematics):
    return build_dataset(tmp_path_factory.mktemp("sharpa"), kinematics)


@pytest.fixture
def calibration(tmp_path):
    """A test-only rig calibration; these values describe no real camera."""

    def camera(parent, mount):
        return {
            "K": [[70.0, 0.0, 47.5], [0.0, 70.0, 47.5], [0.0, 0.0, 1.0]],
            "model": "PINHOLE",
            "distortion": [],
            "resolution": [SIZE, SIZE],
            "rectified": True,
            "parent_link": parent,
            "parent_link_T_cam": mount.tolist(),
        }

    path = tmp_path / "rig.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                RIG: {
                    "head_left": camera(
                        "head_base_link", _mount(OPTICAL, [0.05, 0.0, 0.03])
                    ),
                    "wrist_left": camera(
                        "left_hand_wrist", _mount(np.eye(3), [0, 0, 0.04])
                    ),
                    "wrist_right": camera(
                        "right_arm_Link_7", _mount(OPTICAL, [0.02, 0.0, 0.05])
                    ),
                }
            }
        )
    )
    return path


def _convert(dataset, output, *extra):
    return sharpa_to_zarr.main(
        [
            "--dataset-root",
            str(dataset),
            "--output-dir",
            str(output),
            "--task-name",
            "test_task",
            *extra,
        ]
    )


def _report(path):
    return json.loads(Path(path).read_text())


def test_public_data_path_is_a_valid_structural_sample(dataset, tmp_path):
    out = tmp_path / "out"
    assert _convert(dataset, out, "--episodes", "0-1", "--validate") == 0
    first = out / "2026-01-01-19-04-05-000000.zarr"
    second = out / "2026-01-01-19-04-05-800000.zarr"
    assert first.is_dir() and second.is_dir()
    # The name is the episode's UTC start and round-trips through the registry.
    assert episode_hash_to_timestamp_ms(second.stem) == SEASON_START_NS // 10**6 + 800

    group = zarr.open_group(second, mode="r")
    attrs = group.attrs
    assert attrs["data_status"] == "structural_sample"
    assert attrs["schema_version"] == "v3.1"
    assert attrs["embodiment"] == "dexmate_bimanual"
    assert attrs["morphology"]["platform"] == "dexmate_nth_poc1"
    assert attrs["timestamp_source"] == "synthesized_from_season_name_and_frame_index"
    assert attrs["vendor_recording_id"].endswith(f"{SEASON}/1")
    assert attrs["total_frames"] == LENGTHS[1]
    assert "calibration" not in attrs and "obs_head_pose" not in group
    assert {k for k in group.array_keys() if k.startswith("images.")} == {
        "images.front_1",
        "images.left_wrist",
        "images.right_wrist",
    }
    assert not any(
        "tactile" in k or "relative_timestamp" in k for k in group.array_keys()
    )
    stamps = group["obs_rgb_timestamps_ns"][: LENGTHS[1]]
    expected = SEASON_START_NS + (LENGTHS[0] * 10**9) // FPS
    expected += (np.arange(LENGTHS[1], dtype=np.int64) * 10**9) // FPS
    np.testing.assert_array_equal(stamps, expected)
    features = attrs["features"]
    spec_names = features["right.cmd_hand_joints"]["joint_names"]
    assert spec_names[0] == "right_thumb_CMC_FE" and len(spec_names) == 22
    assert features["obs_aux_joints"]["joint_names"][-1] == "neck_joint_2"

    report = _report(out / f"{second.stem}.validation.json")
    assert report["ok"], report
    assert validate.main([str(first), str(second)]) == 0
    sidecar = _report(out / f"{second.stem}.conversion.json")
    reasons = sidecar["attributes"]["conversion_report"]["data_status_reasons"]
    assert any("calibration" in r for r in reasons)
    assert any("synthesized" in r for r in reasons)


def test_complete_path_passes_the_delivery_profile(
    dataset, calibration, kinematics, tmp_path
):
    out = tmp_path / "out"
    args = ["--episodes", "1", "--calibration", str(calibration)]
    args += ["--utc-timestamp-column", UTC_COLUMN, "--validate", "--save-mp4"]
    assert _convert(dataset, out, *args) == 0
    [path] = list(out.glob("*.zarr"))
    group = zarr.open_group(path, mode="r")
    attrs = group.attrs
    assert attrs["data_status"] == "complete", attrs["conversion_report"]
    assert attrs["timestamp_source"] == f"vendor_column:{UTC_COLUMN}"
    assert (out / f"{path.stem}.mp4").is_file()

    source = load_episode(open_season(dataset / SEASON, "v2.1"), 1)
    np.testing.assert_array_equal(
        group["obs_rgb_timestamps_ns"][: LENGTHS[1]], source.columns[UTC_COLUMN]
    )
    first_ms = int(source.columns[UTC_COLUMN][0]) // 10**6
    assert path.stem == timestamp_ms_to_episode_hash(first_ms)

    # The optical pose is FK(head_base_link) @ mount, checked per frame.
    platform = load_urdf(kinematics.resolved.platform.asset_path())
    head = np.asarray(group["obs_head_pose"][: LENGTHS[1]])
    observed = source.columns["observation.state"].astype(float)
    for t in (0, LENGTHS[1] - 1):
        values = dict(zip(kinematics.column_names, observed[t]))
        expected = platform.link_transforms(values)["head_base_link"] @ _mount(
            OPTICAL, [0.05, 0.0, 0.03]
        )
        np.testing.assert_allclose(
            _xyzwxyz_to_matrix(head[t : t + 1])[0], expected, atol=1e-9
        )
    cameras = attrs["calibration"]["cameras"]
    assert set(cameras) == {"front_1", "left_wrist", "right_wrist"}
    assert not any("ref_T_cam" in camera for camera in cameras.values())
    assert attrs["camera_mounts"]["left_wrist"]["parent_link"] == "left_hand_wrist"
    assert attrs["intrinsics"]["front_1"][0][0] == 70.0

    flags = ["--schema-version", "--data-status", "--calibration-block"]
    flags += ["--calibration-present", "--camera-coverage", "--rgb-timestamps"]
    assert validate.main([str(path), *flags, "--ego-overlay"]) == 0


def test_joint_limit_conflicts_are_reported_not_hidden(
    tmp_path, kinematics, calibration
):
    dataset = build_dataset(tmp_path / "data", kinematics, excursion=0.2)
    raw = tmp_path / "raw"
    assert _convert(dataset, raw, "--episodes", "0", "--validate") == 2
    [path] = list(raw.glob("*.zarr"))
    attrs = zarr.open_group(path, mode="r").attrs
    limits = attrs["conversion_report"]["joint_limits"]
    [record] = limits["source_excursions"]["obs"]
    assert record["joint"] == LIMITED_JOINT
    assert record["array"] == "left.obs_hand_joints"
    assert record["frames_above"] == 3
    assert record["max_above"] == pytest.approx(0.2, abs=1e-5)
    assert (
        not limits["clipped"] and limits["stored_excursions_checked_by_validator"] == 1
    )
    report = validate.validate_episode(path)
    errors = {f.check: f.message for f in report.errors}
    assert "exceed registry limits" in errors["left.obs_hand_joints"]

    clipped = tmp_path / "clipped"
    args = ["--episodes", "0", "--clip-joint-limits", "--validate"]
    args += ["--calibration", str(calibration), "--utc-timestamp-column", UTC_COLUMN]
    assert _convert(dataset, clipped, *args) == 0
    [path] = list(clipped.glob("*.zarr"))
    group = zarr.open_group(path, mode="r")
    limits = group.attrs["conversion_report"]["joint_limits"]
    assert limits["clipped"] and limits["stored_excursions_checked_by_validator"] == 0
    assert limits["source_excursions"]["obs"][0]["joint"] == LIMITED_JOINT
    assert group.attrs["data_status"] == "structural_sample"
    column = kinematics.resolved.end_effectors["left"].joint_names.index(LIMITED_JOINT)
    stored = group["left.obs_hand_joints"][2:5, column]
    upper = kinematics.limits[kinematics.column_names.index(LIMITED_JOINT), 1]
    np.testing.assert_allclose(stored, upper, atol=1e-6)
    assert validate.validate_episode(path).ok


def test_v30_reader_matches_v21_and_conversions_agree(dataset, tmp_path):
    v21 = open_season(dataset / SEASON, "v2.1")
    v30 = open_season(dataset / SEASON, "v3.0")
    assert (v21.version, v30.version) == ("v2.1", "v3.0")
    for index in range(len(LENGTHS)):
        a, b = load_episode(v21, index), load_episode(v30, index)
        assert a.info.frame_offset == b.info.frame_offset == b.info.dataset_from_index
        assert a.columns.keys() == b.columns.keys()
        for name, values in a.columns.items():
            np.testing.assert_array_equal(values, b.columns[name], err_msg=name)
        for camera in ("head_left", "wrist_right"):
            frames_a, frames_b = list(a.frames(camera)), list(b.frames(camera))
            assert len(frames_a) == len(frames_b) == LENGTHS[index]
            for x, y in zip(frames_a, frames_b):
                assert np.abs(x.astype(int) - y.astype(int)).mean() < 1.0

    outputs = {}
    for version in ("v2.1", "v3.0"):
        out = tmp_path / version
        args = ["--episodes", "1", "--lerobot-version", version, "--max-frames", "12"]
        assert _convert(dataset, out, *args) == 0
        [outputs[version]] = list(out.glob("*.zarr"))
    assert outputs["v2.1"].name == outputs["v3.0"].name
    a, b = (zarr.open_group(p, mode="r") for p in outputs.values())
    assert a.attrs["total_frames"] == b.attrs["total_frames"] == 12
    assert b.attrs["source_lerobot_version"] == "v3.0"
    for key in a.array_keys():
        if not key.startswith("images.") and key != "annotations":
            np.testing.assert_array_equal(a[key][:], b[key][:], err_msg=key)


def test_names_are_deterministic_and_collisions_fail_clearly(dataset, tmp_path, caplog):
    out = tmp_path / "out"
    assert _convert(dataset, out, "--episodes", "0", "--max-frames", "4") == 0
    before = sorted(p.name for p in out.iterdir())
    assert _convert(dataset, out, "--episodes", "0", "--max-frames", "4") == 1
    assert "pass --overwrite" in caplog.text
    assert (
        _convert(dataset, out, "--episodes", "0", "--max-frames", "4", "--overwrite")
        == 0
    )
    assert sorted(p.name for p in out.iterdir()) == before

    # A second rig whose season started at the same second shares every name.
    twin = tmp_path / "twin"
    shutil.copytree(dataset / SEASON, twin / SEASON)
    shutil.copytree(dataset / SEASON, twin / SEASON.replace(RIG, "POC99002"))
    crowded = tmp_path / "crowded"
    assert _convert(twin, crowded, "--episodes", "0", "--max-frames", "4") == 1
    assert "is shared by" in caplog.text
    assert not crowded.exists() or not any(crowded.iterdir())


def test_frame_count_mismatch_fails_loudly(tmp_path, kinematics, caplog):
    dataset = build_dataset(tmp_path / "data", kinematics)
    video = (
        dataset
        / SEASON
        / "lerobotv2.1/videos/chunk-000/observation.images.wrist_left/episode_000000.mp4"
    )
    _write_video(video, [_frame(g, 1) for g in range(LENGTHS[0] - 2)])
    assert _convert(dataset, tmp_path / "out", "--episodes", "0") == 1
    assert "decoded 22 frames, parquet has 24 rows" in caplog.text
    assert not list((tmp_path / "out").glob("*.zarr"))
    assert not list((tmp_path / "out").glob(".*partial"))


def test_the_template_and_malformed_calibrations_are_rejected(
    kinematics, calibration, tmp_path
):
    template = Path(sharpa_to_zarr.__file__).with_name("calibration_template.yaml")
    with pytest.raises(ConversionError, match="placeholder"):
        load_calibration_file(template, kinematics)
    good = yaml.safe_load(calibration.read_text())
    assert set(load_calibration_file(calibration, kinematics)[RIG]) == {
        "head_left",
        "wrist_left",
        "wrist_right",
    }
    for change, message in (
        ({"parent_link_T_cam": (2 * np.eye(4)).tolist()}, "rigid"),
        ({"parent_link": "camera_link"}, "not a link"),
        ({"model": "FISHEYE"}, "unknown camera model"),
        ({"focal": 1.0}, "unknown field"),
    ):
        broken = {RIG: {"head_left": {**good[RIG]["head_left"], **change}}}
        path = tmp_path / "broken.yaml"
        path.write_text(yaml.safe_dump(broken))
        with pytest.raises(ConversionError, match=message):
            load_calibration_file(path, kinematics)
