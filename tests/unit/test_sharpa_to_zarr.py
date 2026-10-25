"""Convert tiny synthetic SharpaDex seasons end to end on main's writer."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import simplejpeg
import zarr
from sharpadex_release import (
    LENGTHS,
    LICENSE,
    PER_HAND_TEXT,
    SEASON,
    TAGS,
    TASK,
    build_season,
    deform,
    records,
    write_manifest,
    write_readme,
    write_video,
)

from egomimic.rldb.zarr import dexmate_rules, validate
from egomimic.rldb.zarr.episode_rules import episode_hash_start_ms
from egomimic.scripts.sharpa_process import lerobot_reader, sharpa_to_zarr
from egomimic.scripts.sharpa_process.kinematics import load_spec
from egomimic.scripts.sharpa_process.lerobot_reader import load_episode, open_season
from egomimic.scripts.sharpa_process.sharpa_to_zarr import (
    LAB,
    OPERATOR,
    REPORT_FORMAT,
    ConversionError,
    encode_tactile_images,
    episode_hash,
    episode_segments,
    read_report,
    split_tactile,
    verification_report,
)
from egomimic.scripts.sharpa_process.sharpa_tracks import SharpaKinematics
from egomimic.scripts.sharpa_process.sharpadex_records import (
    read_manifest,
    vendor_recording_id,
)

OTHER_TASK = "fold_towel"
SEASON_B = SEASON.replace("01_02", "01_03")


@pytest.fixture(scope="module")
def kinematics():
    return SharpaKinematics()


def _release(root: Path, kinematics, **season) -> tuple[Path, Path]:
    release = root / "release"
    write_readme(release)
    build_season(release, kinematics, **season)
    rows = records(
        task=season.get("task", TASK),
        season=season.get("season", SEASON),
        lengths=season.get("lengths", LENGTHS),
    )
    return release, write_manifest(root / "manifest.jsonl", release, rows)


def _convert(release, manifest, output, *extra):
    return sharpa_to_zarr.main(
        [
            "--dataset-root",
            str(release),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            *extra,
        ]
    )


def _json(path: Path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def converted(tmp_path_factory, kinematics):
    root = tmp_path_factory.mktemp("sharpadex")
    release, manifest = _release(root, kinematics)
    out = root / "out"
    assert _convert(release, manifest, out, "--include-head-right") == 0
    return release, manifest, out


def _store(out: Path, index: int):
    name = records()[index].episode_hash
    return out / f"{name}.zarr", zarr.open_group(out / f"{name}.zarr", mode="r")


def test_an_episode_is_written_on_main_s_contract(converted):
    release, _, out = converted
    path, group = _store(out, 1)
    attrs = group.attrs
    assert attrs["embodiment"] == "dexmate_bimanual"
    assert attrs["schema_version"] == 1  # the writer's, never an override
    assert attrs["task_name"] == TASK
    assert attrs["task_description"] == TAGS["task_instruction"]
    assert (
        (attrs["operator"], attrs["lab"]) == (OPERATOR, LAB) == ("sharpa", "sharpa_dex")
    )
    assert attrs["scene"] is attrs["objects"] is attrs["intrinsics"] is None
    assert "intrinsics" in attrs
    assert attrs["vendor_recording_id"] == vendor_recording_id(TASK, SEASON, 1)
    assert attrs["source_episode_id"] == f"{TASK}/{SEASON}/1"
    assert attrs["license"] == LICENSE
    assert attrs["source_revision"] == "abc123" and attrs["hash_offset_ms"] == 0
    report = attrs["conversion_report"]
    assert report["data_status"] == "complete", report["data_status_reasons"]
    assert any("calibration" in x for x in report["data_limitations"])
    assert report["tcp_check"]["passed"]
    names = attrs["sharpa_array_names"]
    assert names["right.obs_hand_joints"] == list(
        dexmate_rules.HAND_JOINT_NAMES["right"]
    )
    assert names["obs_aux_joints"] == list(dexmate_rules.AUX_JOINT_NAMES)
    assert names["left.obs_tactile"][6] == "index_fx"
    assert attrs["sharpa_tactile_units"]["tz"] == "N*m"
    # The writer's features stay the writer's.
    assert attrs["features"]["left.obs_joints"]["names"] == ["dim_0"]
    np.testing.assert_array_equal(
        group["obs_rgb_timestamps_ns"][:3],
        records()[1].start_ns + (np.arange(3) * 10**9) // 30,
    )
    for key in (
        "left.cmd_hand_keypoints",
        dexmate_rules.HEAD_LINK_POSE,
        "sharpa.obs_tcp_pose",
        "images.head_right",
        "images.left_wrist",
    ):
        assert key in group, key
    assert (out / f"{path.stem}.mp4").stat().st_size > 0
    sidecar = _json(out / f"{path.stem}.conversion.json")
    assert sidecar["vendor_recording_id"] == attrs["vendor_recording_id"]
    assert sidecar["source_episode_id"] == attrs["source_episode_id"]


def test_episodes_pass_main_s_validator_with_only_the_expected_waivers(converted):
    _, _, out = converted
    paths = sorted(str(p) for p in out.glob("*.zarr"))
    assert len(paths) == 2
    assert validate.main([*paths, "--allow-missing"]) == 0
    # Without --allow-missing, scene/objects/intrinsics fail as designed.
    assert validate.main(paths) == 1
    for path in paths:
        report, problems = read_report(Path(path).with_suffix(".validation.json"))
        assert problems == [], problems
        assert report["waived"] == ["intrinsics", "objects", "scene"]
        assert max(report["checks"]["fk_residual"]["max_m"].values()) < 1e-9


def test_an_episode_stages_like_any_vendor_episode(converted, monkeypatch):
    # Main's stager reads zarr.json from S3; serve it from disk instead.
    import io
    import sys
    from types import SimpleNamespace

    from egomimic.scripts.backfill_scripts import stage_processed_folder

    _, _, out = converted
    path, _ = _store(out, 0)
    prefix = f"processed_v3/{LAB}/{path.name}/"

    class FakeS3:
        def get_object(self, Bucket, Key):
            assert (Bucket, Key) == ("rldb", prefix + "zarr.json")
            return {"Body": io.BytesIO((path / "zarr.json").read_bytes())}

    fake = SimpleNamespace(client=lambda *args, **kwargs: FakeS3())
    monkeypatch.setitem(sys.modules, "boto3", fake)
    cfg = {"endpoint": None, "key": None, "secret": None}
    [row] = stage_processed_folder.read_batch([prefix], LAB, cfg)
    assert row["episode_hash"] == path.stem
    assert row["embodiment"] == "dexmate_bimanual" and row["task"] == TASK
    assert row["num_frames"] == LENGTHS[0] and row["fps"] == 30
    assert row["has_annotations"] and row["created_at"].startswith(
        "2026-01-01T19:04:05"
    )
    # raw_attrs becomes JSONB: plain JSON, no NaN.
    attrs = json.loads(row["raw_attrs"])
    assert attrs["license"] == LICENSE and attrs["lab"] == LAB
    assert "NaN" not in row["raw_attrs"] and "Infinity" not in row["raw_attrs"]


def test_both_hands_annotations_and_skills(converted):
    _, _, out = converted

    def spans(group, key):
        return [json.loads(bytes(x)) for x in group[key][:]]

    _, first = _store(out, 0)
    assert spans(first, "annotations") == [
        {"text": "Grip the iron head.", "start_idx": 0, "end_idx": 10},
        {"text": "Iron the garment.", "start_idx": 10, "end_idx": LENGTHS[0]},
    ]
    assert [s["text"] for s in spans(first, "annotations_skill")] == ["grip", "press"]
    _, second = _store(out, 1)
    joined = PER_HAND_TEXT.replace(" | ", "; ")
    assert spans(second, "annotations")[0] == {
        "text": joined,
        "start_idx": 0,
        "end_idx": 10,
    }
    assert "Right hand: reach: Moves down." in joined
    assert [s["text"] for s in spans(second, "annotations_skill")] == ["press"]
    notes = second.attrs["conversion_report"]["annotations"]["notes"]
    assert notes == ["1 span(s) joined at ' | ' with '; '"]


def test_per_hand_subtask_text_is_joined_not_cut():
    # play_tic_tac_toe and 7 other tasks describe both hands in one span.
    row = {
        "segments": [
            {
                "label": "subtasks",
                "start_step": 0,
                "end_step": 4,
                "id": "reasoning-language-00001",
                "text": PER_HAND_TEXT,
            },
            {
                "label": "subtasks",
                "start_step": 5,
                "end_step": 12,
                "id": "subtask-002",
                "text": "Iron the garment. | Skill: press | extra",
            },
            {"label": "subtasks", "start_step": 3, "end_step": 1, "id": "bad"},
            {"label": "other", "start_step": 0, "end_step": 1, "text": "x"},
        ]
    }
    subtasks, skills, notes = episode_segments(row, 10)
    assert subtasks == [
        ("Left hand: idle: Stays still.; Right hand: reach: Moves down.", 0, 5),
        ("Iron the garment.", 5, 10),  # clipped to the retained frames
    ]
    assert skills == [("press", 5, 10)]
    assert notes == [
        "skipped malformed span bad",
        "1 span(s) joined at ' | ' with '; '",
    ]
    for text, _, _ in subtasks + skills:
        assert " | " not in text


def test_tactile_columns_and_strips_keep_the_layout(converted):
    release, _, out = converted
    _, group = _store(out, 1)
    source = load_episode(open_season(release / TASK / SEASON), 1)
    tactile = source.columns["observation.tactile"]
    np.testing.assert_array_equal(
        group["left.obs_tactile"][: LENGTHS[1]], tactile[:, :30]
    )
    np.testing.assert_array_equal(
        group["right.obs_tactile"][: LENGTHS[1]], tactile[:, 30:]
    )
    assert group.attrs["features"]["right.obs_tactile_image"]["shape"] == [240, 1200, 3]
    strip = simplejpeg.decode_jpeg(bytes(group["right.obs_tactile_image"][0:1][0]))
    assert strip.shape == (240, 1200, 3)
    # Row 1 of the mosaic is the right hand; tiles run thumb to little.
    expected = deform(LENGTHS[0])[240:, :, 0]
    assert np.abs(strip[..., 0].astype(int) - expected).max() <= 3
    assert np.abs(strip.astype(int) - strip[..., :1]).max() <= 2
    with pytest.raises(ConversionError, match="shape"):
        split_tactile(np.zeros((3, 59)))


class _FakeEpisode:
    def __init__(self, frames):
        self._frames, self.length = frames, len(frames)

    def frames(self, camera, threads=None):
        assert camera == "tactile_deform"
        return iter(self._frames)


def test_tactile_strips_refuse_a_colour_or_misshapen_mosaic():
    good = deform(0)
    strips = encode_tactile_images(_FakeEpisode([good]), 90)
    assert set(strips) == {"left.obs_tactile_image", "right.obs_tactile_image"}
    left = simplejpeg.decode_jpeg(bytes(strips["left.obs_tactile_image"][0][0]))
    assert abs(int(left[120, 120, 0]) - int(good[120, 120, 0])) <= 2  # thumb tile
    colour = good.copy()
    colour[..., 1] = 0
    with pytest.raises(ConversionError, match="not grayscale"):
        encode_tactile_images(_FakeEpisode([colour]), 90)
    with pytest.raises(ConversionError, match="tactile_deform frame 0"):
        encode_tactile_images(_FakeEpisode([good[:, :960]]), 90)


def test_decoders_get_a_share_of_the_cpus(tmp_path, kinematics, monkeypatch):
    release, manifest = _release(tmp_path, kinematics, lengths=(12,))
    seen = []
    decode = lerobot_reader.decode_frames

    def spy(path, *, expected, threads=None):
        seen.append((Path(path).parent.name, threads))
        return decode(path, expected=expected, threads=threads)

    previews = []
    init = sharpa_to_zarr._Preview.__init__

    def preview_spy(self, path, fps, threads=None):
        previews.append(threads)
        init(self, path, fps, threads)

    monkeypatch.setattr(lerobot_reader, "decode_frames", spy)
    monkeypatch.setattr(sharpa_to_zarr._Preview, "__init__", preview_spy)
    monkeypatch.setattr(sharpa_to_zarr, "available_cpus", lambda: 11)
    out = tmp_path / "out"
    assert _convert(release, manifest, out, "--include-head-right") == 0
    # 4 cameras + tactile_deform decode at once: 11 // 5 = 2 threads each.
    assert sorted(seen) == sorted(
        (f"observation.images.{c}", 2)
        for c in (
            "head_left",
            "head_right",
            "wrist_left",
            "wrist_right",
            "tactile_deform",
        )
    )
    assert previews == [2]


def test_a_swapped_tcp_layout_makes_a_structural_sample(tmp_path, kinematics):
    release, manifest = _release(tmp_path, kinematics, lengths=(12,), swap_tcp=True)
    out = tmp_path / "out"
    assert _convert(release, manifest, out) == 2
    [path] = list(out.glob("*.zarr"))
    report = zarr.open_group(path, mode="r").attrs["conversion_report"]
    assert report["data_status"] == "structural_sample"
    assert not report["tcp_check"]["passed"]
    assert any("TCP check failed" in r for r in report["data_status_reasons"])
    verification, problems = read_report(path.with_suffix(".validation.json"))
    assert problems
    assert any(e.startswith("tcp_check: ") for e in verification["errors"])
    assert any(e.startswith("conversion: data_status") for e in verification["errors"])


def test_gross_excursions_fail_validation_and_make_a_structural_sample(
    tmp_path, kinematics
):
    release, manifest = _release(
        tmp_path, kinematics, lengths=(12,), excursion=("left_middle_DIP", 0.3)
    )
    out = tmp_path / "out"
    assert _convert(release, manifest, out) == 2
    [path] = list(out.glob("*.zarr"))
    report = zarr.open_group(path, mode="r").attrs["conversion_report"]
    [record] = report["joint_limits"]["source_excursions"]["obs"]
    assert record["joint"] == "left_middle_DIP" and record["validator_check"] == "error"
    assert report["data_status"] == "structural_sample"
    _, problems = read_report(path.with_suffix(".validation.json"))
    assert any("left.obs_hand_joints" in p for p in problems)


def test_small_excursions_are_warnings_and_stay_complete(tmp_path, kinematics):
    release, manifest = _release(
        tmp_path, kinematics, lengths=(12,), excursion=("left_middle_DIP", 0.1)
    )
    out = tmp_path / "out"
    assert _convert(release, manifest, out) == 0
    [path] = list(out.glob("*.zarr"))
    report = zarr.open_group(path, mode="r").attrs["conversion_report"]
    assert report["data_status"] == "complete"
    assert any(
        "inside the validator's error margin" in x for x in report["data_limitations"]
    )
    verification, problems = read_report(path.with_suffix(".validation.json"))
    assert problems == []
    assert any("left_middle_DIP" in w for w in verification["warnings"])


def test_fk_residual_and_id_errors_reach_the_report(converted, tmp_path, kinematics):
    _, _, out = converted
    path, _ = _store(out, 0)
    copy = tmp_path / path.name
    shutil.copytree(path, copy)
    group = zarr.open_group(copy, mode="a")
    keypoints = group["right.obs_hand_keypoints"][:]
    keypoints[:, 60:63] += 0.02  # pinky tip moved 2 cm off the joints' FK
    group["right.obs_hand_keypoints"][:] = keypoints
    group.attrs["source_episode_id"] = f"{TASK}/{SEASON}/1"
    tcp = group.attrs["conversion_report"]["tcp_check"]
    report = verification_report(copy, kinematics, tcp)
    assert not report["ok"]
    assert any(
        e.startswith("fk_residual: right.obs_hand_keypoints") for e in report["errors"]
    )
    assert any(e.startswith("ids: ") for e in report["errors"])
    missing = verification_report(copy, kinematics, None)
    assert "tcp_check: missing" in missing["errors"]


def test_incomplete_conversions_fail_the_report(converted, tmp_path, kinematics):
    _, _, out = converted
    path, group = _store(out, 0)
    good = verification_report(
        path, kinematics, group.attrs["conversion_report"]["tcp_check"]
    )
    assert good["ok"], good["errors"]
    assert good["checks"]["conversion"] == {
        "data_status": "complete",
        "data_status_reasons": [],
        "source_frames": LENGTHS[0],
        "retained_frames": LENGTHS[0],
        "stored_frames": LENGTHS[0],
    }
    for change, message in (
        (
            {"source": {"source_frames": LENGTHS[0], "retained_frames": 4}},
            "frame counts differ",
        ),
        (
            {"data_status": "structural_sample", "data_status_reasons": ["x"]},
            "data_status",
        ),
        ({"source": {}}, "frame counts differ"),
    ):
        copy = tmp_path / f"{len(list(tmp_path.iterdir()))}" / path.name
        shutil.copytree(path, copy)
        store = zarr.open_group(copy, mode="a")
        store.attrs["conversion_report"] = {
            **store.attrs["conversion_report"],
            **change,
        }
        tcp = store.attrs["conversion_report"]["tcp_check"]
        report = verification_report(copy, kinematics, tcp)
        assert not report["ok"]
        assert any(e.startswith("conversion: " + message) for e in report["errors"])


def test_episodes_are_always_converted_whole(converted, tmp_path):
    release, manifest, _ = converted
    with pytest.raises(SystemExit):
        _convert(release, manifest, tmp_path / "out", "--max-frames", "4")


def _report_case(good, change):
    """``good`` with ``change`` applied: {"a/b/c": value}; None deletes."""
    report = json.loads(json.dumps(good))
    for path, value in change.items():
        *parents, key = path.split("/")
        node = report
        for parent in parents:
            node = node[parent]
        if value is None:
            node.pop(key)
        else:
            node[key] = value
    return report


@pytest.fixture(scope="module")
def good_report(converted):
    _, _, out = converted
    path, group = _store(out, 0)
    report = _json(path.with_suffix(".validation.json"))
    assert report["ok"] and report["errors"] == []
    return path, group, report


def test_a_good_report_reads_clean(good_report, tmp_path):
    path, group, good = good_report
    copy = tmp_path / f"{path.stem}.validation.json"
    copy.write_text(json.dumps(good))
    vid, sid = group.attrs["vendor_recording_id"], group.attrs["source_episode_id"]
    assert read_report(copy, vid, sid) == (good, [])
    assert "is for" in read_report(copy, "other")[1][0]
    assert "is for" in read_report(copy, vid, f"{TASK}/{SEASON}/1")[1][0]
    # A report read under another episode's name is not that episode's.
    other = tmp_path / "2026-01-01-00-00-00-000000.validation.json"
    other.write_text(json.dumps(good))
    assert "describes" in read_report(other)[1][0]


@pytest.mark.parametrize(
    "change, message",
    [
        # Missing fields: the aggregate alone is not enough.
        ({"checks": None}, "missing"),
        ({"source_episode_id": None}, "missing"),
        ({"checks/conversion": None}, "missing"),
        ({"checks/tcp_check": None}, "missing"),
        # Ids that do not correspond.
        ({"source_episode_id": f"{TASK}/{SEASON}/1"}, "do not correspond"),
        # Failed checks under ok=True, errors=[].
        ({"checks/validator/ok": False, "checks/validator/errors": 2}, "validator"),
        ({"checks/tcp_check/passed": False}, "tcp_check"),
        ({"checks/tcp_check/sides/left/rotation_error_deg_median": 5.0}, "tcp_check"),
        (
            {"checks/tcp_check/sides/right/swapped_rotation_error_deg_median": 0.0},
            "tcp_check",
        ),
        ({"checks/tcp_check/sides/left": None}, "tcp_check"),
        ({"checks/fk_residual/max_m/right.cmd": 0.02}, "fk_residual"),
        ({"checks/fk_residual/max_m/right.cmd": None}, "fk_residual"),
        ({"checks/fk_residual/max_m/left.obs": "0"}, "fk_residual"),
        ({"checks/conversion/data_status": "structural_sample"}, "conversion"),
        ({"checks/conversion/retained_frames": 3}, "conversion"),
        ({"waived": ["scene", "task_name"]}, "waived"),
    ],
)
def test_reports_with_failed_or_missing_checks_fail(
    good_report, tmp_path, change, message
):
    path, _, good = good_report
    copy = tmp_path / f"{path.stem}.validation.json"
    copy.write_text(json.dumps(_report_case(good, change)))
    _, problems = read_report(copy)
    assert problems, change
    assert any(message in p for p in problems), problems


def test_missing_or_malformed_reports_fail(good_report, tmp_path):
    path, _, good = good_report
    report = tmp_path / f"{path.stem}.validation.json"

    def problems(value):
        report.write_text(value if isinstance(value, str) else json.dumps(value))
        return read_report(report)[1]

    assert problems("{cut off") and "unreadable" in problems("{cut off")[0]
    nan = json.dumps(good).replace(
        '"fk_residual": {"max_m": {"left.obs": ',
        '"fk_residual": {"max_m": {"left.obs": NaN, "x": ',
        1,
    )
    assert "NaN" in nan and "unreadable" in problems(nan)[0]
    assert "not a" in problems({**good, "format": "other"})[0]
    assert "not a" in problems([good])[0]
    assert "lists of strings" in problems({**good, "errors": "none"})[0]
    assert "missing" in problems({k: v for k, v in good.items() if k != "warnings"})[0]
    assert "disagrees" in problems({**good, "ok": False})[0]
    assert "disagrees" in problems({**good, "ok": True, "errors": ["x"]})[0]
    assert "disagrees" in problems({**good, "ok": "yes"})[0]
    assert "1 validation error" in problems({**good, "ok": False, "errors": ["x"]})[-1]
    report.unlink()
    assert read_report(report) == (None, [f"no {report.name}"])


def test_episode_names_round_trip_through_main_s_hash_parser():
    start_ns = records()[0].start_ns + 123_456_789
    name = episode_hash(start_ns)
    assert episode_hash_start_ms(name) == start_ns // 10**6
    assert name == "2026-01-01-19-04-05-123000"


def test_frame_count_mismatch_fails_loudly(tmp_path, kinematics, caplog):
    release, manifest = _release(tmp_path, kinematics, lengths=(12,))
    video = (
        release
        / TASK
        / SEASON
        / "lerobot_v2.1/videos/chunk-000/observation.images.wrist_left/episode_000000.mp4"
    )
    write_video(video, [np.zeros((32, 32, 3), np.uint8)] * 10)
    out = tmp_path / "out"
    assert _convert(release, manifest, out) == 1
    assert "decoded 10 frames, parquet has 12 rows" in caplog.text
    assert not list(out.glob("*.zarr")) and not list(out.glob(".*partial"))


def test_absent_cameras_and_layout_changes_are_reported(tmp_path, kinematics):
    release, manifest = _release(
        tmp_path, kinematics, lengths=(12,), cameras=("head_left", "wrist_left")
    )
    out = tmp_path / "out"
    assert _convert(release, manifest, out, "--include-head-right") == 0
    [path] = list(out.glob("*.zarr"))
    cameras = zarr.open_group(path, mode="r").attrs["conversion_report"]["cameras"]
    assert cameras["absent_in_source"] == ["head_right", "wrist_right"]
    assert cameras["stored"] == {"front_1": "head_left", "left_wrist": "wrist_left"}

    info_path = release / TASK / SEASON / "lerobot_v2.1/meta/info.json"
    info = _json(info_path)
    names = info["features"]["observation.tactile"]["names"]
    names[0], names[1] = names[1], names[0]
    info_path.write_text(json.dumps(info))
    with pytest.raises(ConversionError, match="unexpected SharpaDex layout"):
        sharpa_to_zarr.check_vendor_layout(open_season(release / TASK / SEASON))


def test_the_manifest_selects_and_the_licence_is_required(tmp_path, kinematics, caplog):
    caplog.set_level(logging.INFO)
    release, manifest = _release(tmp_path, kinematics)
    header, rows = read_manifest(manifest)
    rows[f"{TASK}/{SEASON}/0"].decision = "exclude"
    rows[f"{TASK}/{SEASON}/0"].reasons = ["corrupt frame"]
    edited = write_manifest(tmp_path / "edited.jsonl", release, list(rows.values()))
    out = tmp_path / "out"
    assert _convert(release, edited, out, "--episodes", "0-1") == 0
    assert len(list(out.glob("*.zarr"))) == 1
    assert "skipped" in caplog.text and "corrupt frame" in caplog.text

    # The licence is fixed; the release README must agree with it.
    write_readme(release, license=None)
    out = tmp_path / "no_license"
    assert _convert(release, manifest, out, "--episodes", "0") == 1
    assert "no license" in caplog.text
    with pytest.raises(SystemExit):
        _convert(release, manifest, out, "--episodes", "0", "--license", LICENSE)
    write_readme(release, license="cc-by-4.0")
    caplog.clear()
    assert _convert(release, manifest, out, "--episodes", "0") == 1
    assert "'cc-by-4.0' conflicts with cc-by-nc-4.0" in caplog.text
    write_readme(release, license="CC-BY-NC-4.0")
    assert _convert(release, manifest, out, "--episodes", "0") == 0
    [path] = list(out.glob("*.zarr"))
    assert zarr.open_group(path, mode="r").attrs["license"] == "cc-by-nc-4.0"
    assert sharpa_to_zarr.LICENSE == LICENSE == "cc-by-nc-4.0"

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        manifest.read_text().replace(f"{TASK}/{SEASON}/1", f"{TASK}/{SEASON}/9")
    )
    with pytest.raises(SystemExit, match="id problem"):
        _convert(release, bad, out)


# ---------------------------------------------------------------------------
# Selecting by recording id and resuming
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def three_seasons(tmp_path_factory, kinematics):
    """``TASK/SEASON``, ``TASK/SEASON_B`` and ``SEASON`` again under ``OTHER_TASK``."""
    root = tmp_path_factory.mktemp("sharpadex_ids")
    release = root / "release"
    write_readme(release)
    rows = []
    for i, (task, season) in enumerate(
        ((TASK, SEASON), (TASK, SEASON_B), (OTHER_TASK, SEASON))
    ):
        build_season(release, kinematics, task=task, season=season, lengths=(6, 6))
        start = records()[0].start_ns + i * 10**12
        rows += records(task=task, season=season, lengths=(6, 6), start_ns=start)
    return release, write_manifest(root / "manifest.jsonl", release, rows)


def _ids_file(path: Path, lines) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines))
    return path


def _mtimes(folder: Path) -> dict[str, int]:
    return {str(p.relative_to(folder)): p.stat().st_mtime_ns for p in folder.rglob("*")}


def test_recording_ids_convert_exactly_the_listed_episodes(
    three_seasons, tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    release, manifest = three_seasons
    _, rows = read_manifest(manifest)
    wanted = [
        f"{TASK}/{SEASON_B}/1",
        f"{OTHER_TASK}/{SEASON}/0",
        f"{TASK}/{SEASON_B}/0",
    ]
    lines = ["# chunk_0000", "", f"  {wanted[0]}  ", wanted[1], wanted[0], wanted[2]]
    ids = _ids_file(tmp_path / "ids.txt", lines)

    def no_scan(*args, **kwargs):
        raise AssertionError("--recording-ids must not scan the release")

    monkeypatch.setattr(sharpa_to_zarr, "discover_task_seasons", no_scan)
    out = tmp_path / "out"
    assert _convert(release, manifest, out, "--recording-ids", str(ids)) == 0
    # TASK/SEASON shares a season name with OTHER_TASK/SEASON; not opened.
    assert "3 episode(s) from 2 season(s)" in caplog.text
    owners = {
        path.stem: zarr.open_group(path, mode="r").attrs["vendor_recording_id"]
        for path in out.glob("*.zarr")
    }
    assert owners == {rows[r].episode_hash: rows[r].vendor_recording_id for r in wanted}
    other = zarr.open_group(out / f"{rows[wanted[1]].episode_hash}.zarr", mode="r")
    assert other.attrs["task_name"] == other.attrs["source_task_folder"] == OTHER_TASK
    assert other.attrs["source_episode_id"] == wanted[1]


def test_recording_ids_must_be_kept_manifest_episodes(three_seasons, tmp_path):
    release, manifest = three_seasons
    _, rows = read_manifest(manifest)
    held = f"{TASK}/{SEASON_B}/1"
    rows[held].decision = "hold_gross_excursion"
    edited = write_manifest(tmp_path / "edited.jsonl", release, list(rows.values()))
    good = f"{TASK}/{SEASON}/0"
    out = tmp_path / "out"

    def run(lines, *extra):
        ids = _ids_file(tmp_path / "ids.txt", lines)
        return _convert(release, edited, out, "--recording-ids", str(ids), *extra)

    with pytest.raises(SystemExit, match=r"1 recording id\(s\).*nope/s/0 \(not in"):
        run([good, "nope/s/0"])
    with pytest.raises(SystemExit, match=re.escape(f"{held} (hold_gross_excursion)")):
        run([good, held])
    with pytest.raises(SystemExit, match=r"12 recording id\(s\).*s/9 .* and 2 more$"):
        run([f"nope/s/{i}" for i in range(12)])
    with pytest.raises(SystemExit, match="drop --tasks"):
        run([good], "--tasks", TASK)
    with pytest.raises(SystemExit, match="drop --seasons, --episodes"):
        run([good], "--seasons", SEASON, "--episodes", "0")
    with pytest.raises(SystemExit, match="lists no recording ids"):
        run(["# nothing yet", ""])
    assert not out.exists()


def test_skip_done_resumes_without_touching_finished_outputs(
    three_seasons, tmp_path, capsys, caplog
):
    caplog.set_level(logging.INFO)
    release, manifest = three_seasons
    _, rows = read_manifest(manifest)
    redo, keep = f"{TASK}/{SEASON_B}/0", f"{OTHER_TASK}/{SEASON}/1"
    redo_hash, keep_hash = rows[redo].episode_hash, rows[keep].episode_hash
    ids = _ids_file(tmp_path / "ids.txt", [keep, redo])
    out = tmp_path / "out"

    def run(*extra):
        capsys.readouterr()
        caplog.clear()
        code = _convert(release, manifest, out, "--recording-ids", str(ids), *extra)
        return code, capsys.readouterr().out

    def kept_outputs(mtimes):
        return {k: v for k, v in mtimes.items() if k.startswith(keep_hash)}

    code, printed = run()
    assert code == 0 and "already done" not in printed
    # Age every output so that any rewrite shows.
    for path in out.rglob("*"):
        os.utime(path, ns=(10**18, 10**18))
    before = _mtimes(out)

    code, printed = run("--skip-done")
    assert code == 0
    assert "Converted 0 episode(s), 0 failed, 2 already done, 0 skipped" in printed
    assert f"already done {redo}" in caplog.text
    assert _mtimes(out) == before

    # A store without its conversion.json is unfinished: redone, given --overwrite.
    (out / f"{redo_hash}.conversion.json").unlink()
    code, _ = run("--skip-done")
    assert code == 1 and "pass --overwrite" in caplog.text
    code, printed = run("--skip-done", "--overwrite")
    assert code == 0
    assert "Converted 1 episode(s), 0 failed, 1 already done, 0 skipped" in printed
    after = _mtimes(out)
    assert after[f"{redo_hash}.zarr"] != before[f"{redo_hash}.zarr"]
    sidecar = _json(out / f"{redo_hash}.conversion.json")
    assert sidecar["vendor_recording_id"] == rows[redo].vendor_recording_id
    assert sidecar["source_episode_id"] == redo
    assert kept_outputs(after) == kept_outputs(before)

    # A cut-off conversion.json (a killed write) is not done either.
    sidecar = out / f"{keep_hash}.conversion.json"
    text = sidecar.read_text()
    sidecar.write_text(text[:20])
    code, printed = run("--skip-done", "--overwrite")
    assert code == 0
    assert "Converted 1 episode(s), 0 failed, 1 already done, 0 skipped" in printed
    assert _json(sidecar)["source_episode_id"] == keep


def test_outputs_owned_by_another_recording_are_refused(
    three_seasons, tmp_path, caplog
):
    release, manifest = three_seasons
    _, rows = read_manifest(manifest)
    keep = f"{OTHER_TASK}/{SEASON}/1"
    name = rows[keep].episode_hash
    ids = _ids_file(tmp_path / "ids.txt", [keep])
    out = tmp_path / "out"

    def run(*extra):
        caplog.clear()
        return _convert(release, manifest, out, "--recording-ids", str(ids), *extra)

    assert run() == 0
    sidecar = out / f"{name}.conversion.json"
    text = sidecar.read_text()
    vid = rows[keep].vendor_recording_id
    other = vendor_recording_id("another_task", SEASON, 1)

    # A sidecar naming another recording is never replaced.
    sidecar.write_text(text.replace(vid, other))
    assert run("--skip-done", "--overwrite") == 1
    assert f"already holds {other}" in caplog.text
    assert other in sidecar.read_text()

    # Canonical id right, source id not corresponding: refused, not done.
    sidecar.write_text(
        text.replace(f'"source_episode_id": "{keep}"', '"source_episode_id": "x/y/1"')
    )
    assert run("--skip-done", "--overwrite") == 1
    assert "source_episode_id 'x/y/1'" in caplog.text

    # A store holding another recording is never replaced, even with --overwrite.
    sidecar.unlink()
    group = zarr.open_group(out / f"{name}.zarr", mode="a")
    group.attrs["vendor_recording_id"] = other
    assert run("--overwrite") == 1
    assert f"already holds {other}" in caplog.text
    assert (
        zarr.open_group(out / f"{name}.zarr", mode="r").attrs["vendor_recording_id"]
        == other
    )


def test_resume_requires_complete_passing_outputs(
    three_seasons, tmp_path, capsys, caplog, monkeypatch
):
    caplog.set_level(logging.INFO)
    release, manifest = three_seasons
    _, rows = read_manifest(manifest)
    sid = f"{OTHER_TASK}/{SEASON}/1"
    name = rows[sid].episode_hash
    ids = _ids_file(tmp_path / "ids.txt", [sid])
    out = tmp_path / "out"
    files = {
        kind: out / f"{name}.{kind}"
        for kind in ("zarr", "conversion.json", "validation.json", "mp4")
    }

    def run(*extra):
        capsys.readouterr()
        caplog.clear()
        code = _convert(release, manifest, out, "--recording-ids", str(ids), *extra)
        return code, capsys.readouterr().out

    def done():
        return f"already done {sid}" in caplog.text

    assert run()[0] == 0
    saved = {k: p.read_bytes() for k, p in files.items() if k != "zarr"}

    def restore():
        for kind, data in saved.items():
            files[kind].write_bytes(data)

    # Matching ids alone are not enough: each output must exist and pass.
    for kind in ("validation.json", "mp4"):
        files[kind].unlink()
        code, printed = run("--skip-done")
        assert code == 1 and not done() and "pass --overwrite" in caplog.text, kind
        restore()
    report = json.loads(saved["validation.json"])
    report["checks"]["tcp_check"]["passed"] = False
    files["validation.json"].write_text(json.dumps(report))
    code, _ = run("--skip-done")
    assert code == 1 and not done()
    restore()
    shutil.move(files["zarr"], tmp_path / "moved.zarr")
    code, _ = run("--skip-done")
    assert code == 1 and not done()
    shutil.move(tmp_path / "moved.zarr", files["zarr"])
    # A store that lost an array since its report passed is not done either.
    lost = files["zarr"] / "left.obs_tactile"
    shutil.move(lost, tmp_path / "lost")
    code, _ = run("--skip-done")
    assert code == 1 and not done() and "left.obs_tactile" in caplog.text
    shutil.move(tmp_path / "lost", lost)
    files["mp4"].write_bytes(b"")
    code, _ = run("--skip-done")
    assert code == 1 and not done()
    restore()
    code, printed = run("--skip-done")
    assert code == 0 and done() and "1 already done" in printed

    # A store holding another recording is refused even beside a matching sidecar.
    other = vendor_recording_id("another_task", SEASON, 1)
    group = zarr.open_group(files["zarr"], mode="a")
    group.attrs["vendor_recording_id"] = other
    code, printed = run("--skip-done", "--overwrite")
    assert code == 1 and f"already holds {other}" in caplog.text
    assert "1 failed" in printed
    group.attrs["vendor_recording_id"] = rows[sid].vendor_recording_id

    # Without --skip-done a sidecar naming another recording is refused too.
    files["conversion.json"].write_text(
        saved["conversion.json"].decode().replace(rows[sid].vendor_recording_id, other)
    )
    code, _ = run("--overwrite")
    assert code == 1 and f"already holds {other}" in caplog.text
    restore()

    # Replacing an output drops its completion marker first, so a crash
    # mid-replacement never leaves it "done".
    def crash(*args, **kwargs):
        raise RuntimeError("killed mid-conversion")

    monkeypatch.setattr(sharpa_to_zarr, "encode_camera", crash)
    code, _ = run("--overwrite")
    assert code == 1 and "killed mid-conversion" in caplog.text
    assert not files["conversion.json"].exists()
    assert not files["validation.json"].exists()
    monkeypatch.undo()
    code, _ = run("--skip-done")
    assert code == 1 and not done() and "pass --overwrite" in caplog.text
    code, printed = run("--skip-done", "--overwrite")
    assert code == 0 and "Converted 1 episode(s), 0 failed" in printed


def test_the_spec_tactile_layout_matches_the_converter():
    for side in dexmate_rules.SIDES:
        tactile = load_spec()["hands"][side]["tactile"]
        assert tuple(tactile["fingers"]) == sharpa_to_zarr.FINGERS
        assert tuple(tactile["channel_names"]) == sharpa_to_zarr.TACTILE_CHANNELS
        assert len(tactile["fingers"]) * len(tactile["channel_names"]) == (
            dexmate_rules.TACTILE_DIM
        )
    assert REPORT_FORMAT == "sharpadex_validation_v1"
