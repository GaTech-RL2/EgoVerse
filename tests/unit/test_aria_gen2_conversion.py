"""Aria Gen 2 conversion: frame-signal matching, demo hashing, per-demo writing
and per-demo SQL rows."""

import datetime
import json
from types import SimpleNamespace

import numpy as np
import pytest

aria_utils = pytest.importorskip("egomimic.scripts.aria_process.aria_utils")
aria_to_zarr = pytest.importorskip("egomimic.scripts.aria_process.aria_to_zarr")
from egomimic.scripts.aria_process.gen2_filter import Gen2FilterConfig  # noqa: E402
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash  # noqa: E402

CFG = Gen2FilterConfig()


def _stamped(ns, **fields):
    return SimpleNamespace(
        tracking_timestamp=datetime.timedelta(microseconds=ns / 1000), **fields
    )


def _pose(ns):
    return _stamped(
        ns, transform_world_device=SimpleNamespace(to_matrix=lambda: np.eye(4))
    )


def _hands(ns, left=True, right=True):
    def hand(x):
        return SimpleNamespace(
            confidence=1.0, landmark_positions_device=[np.array([x, 0.0, 0.5])] * 21
        )

    return _stamped(
        ns,
        left_hand=hand(-0.2) if left else None,
        right_hand=hand(0.2) if right else None,
    )


def test_frame_signals_match_samples_within_tolerance():
    rgb = np.array([0, 33_000_000, 66_000_000, 100_000_000], dtype=np.int64)
    poses = [_pose(t) for t in (0, 33_000_000, 90_000_000, 100_000_000)]
    hands = [_hands(0), _hands(33_000_000, right=False), _hands(66_000_000)]
    gaze = [_stamped(t, combined_gaze_valid=v) for t, v in ((0, True), (66e6, False))]
    sig = aria_utils.gen2_frame_signals(rgb, poses, hands, gaze, np.ones(4, bool), CFG)
    # frame 2: nearest pose is 24 ms away; frame 3: nearest hand is 34 ms away.
    assert sig.tracking_ok.tolist() == [True, True, False, False]
    assert sig.right_tracked.tolist() == [True, False, True, True]
    assert sig.left_tracked.all()
    # frame 1 is 33 ms from both gaze samples: within the 50 ms gaze tolerance.
    assert sig.gaze_valid.tolist() == [True, True, False, False]
    np.testing.assert_allclose(sig.hand_min_dist_m, np.hypot(0.2, 0.5))


def test_demo_hashes_are_unique_and_never_the_parent():
    start_ms = 1794800772000
    parent = timestamp_ms_to_episode_hash(start_ms)
    hashes = aria_to_zarr.gen2_demo_hashes(
        [start_ms * 1_000_000, start_ms * 1_000_000 + 200_000, 1794800830123456789],
        parent_hash=parent,
    )
    assert len(set(hashes)) == 3 and parent not in hashes
    assert hashes[0] == timestamp_ms_to_episode_hash(start_ms + 1)
    assert hashes[1] == timestamp_ms_to_episode_hash(start_ms + 2)
    assert hashes[2] == timestamp_ms_to_episode_hash(1794800830123)


def test_write_demos_one_zarr_per_demo(tmp_path, monkeypatch):
    T = 12
    demo_index = np.array([0] * 5 + [1] * 7)
    feats = {
        "left.obs_ee_pose": np.random.rand(T, 7),
        "right.obs_ee_pose": np.random.rand(T, 7),
        "obs_head_pose": np.random.rand(T, 7),
        "obs_rgb_timestamps_ns": np.arange(T, dtype=np.int64),
        "images.front_1": np.zeros((T, 48, 64, 3), np.uint8),
        "_demo_index": demo_index,
        "_attrs": {"device": "aria_gen2"},
        "_gen2_demos": [
            {
                "demo_index": d,
                "start_s": float(d),
                "end_s": float(d) + 0.5,
                "span_frames": n,
                "error_frames": 0,
                "kept_frames": n,
                "start_ns": 0,
                "end_ns": 0,
                "first_kept_utc_ns": utc,
            }
            for d, n, utc in (
                (0, 5, 1794800772500_000_000),
                (1, 7, 1794800790000_000_000),
            )
        ],
    }
    monkeypatch.setattr(
        aria_to_zarr.AriaVRSExtractor, "process_episode", lambda **kw: dict(feats)
    )
    conv = aria_to_zarr.DatasetConverter(
        raw_path=tmp_path, fps=30, arm="both", convert_mano=False
    )
    meta = dict(task_name="t", operator="o", lab="l", scene="s", objects=["x"])
    results = conv.extract_episode(
        tmp_path / "1794800772000.vrs",
        required_metadata=meta,
        output_dir=tmp_path,
        dataset_name=timestamp_ms_to_episode_hash(1794800772000),
    )
    assert [r["episode_hash"] for r in results] == [
        timestamp_ms_to_episode_hash(1794800772500),
        timestamp_ms_to_episode_hash(1794800790000),
    ]
    for r, n in zip(results, (5, 7)):
        attrs = json.loads((r["zarr_path"] / "zarr.json").read_text())["attributes"]
        assert attrs["total_frames"] == n
        assert attrs["device"] == "aria_gen2"
        assert attrs["demo_count"] == 2
        assert attrs["parent_episode_hash"] == timestamp_ms_to_episode_hash(
            1794800772000
        )


PREFIX = "s3://rldb/processed_v3/aria_gen2"


def _fake_sql(monkeypatch, run_conversion, rows):
    inserted = []
    monkeypatch.setattr(
        run_conversion, "episode_hash_to_table_row", lambda e, h: rows.get(h)
    )
    monkeypatch.setattr(
        run_conversion,
        "episodes_with_path_prefix_or_error_marker",
        lambda e, prefix, marker: [
            r
            for r in rows.values()
            if r.zarr_processed_path.startswith(prefix)
            or marker in r.zarr_processing_error
        ],
    )

    def add(e, r):
        inserted.append(r.episode_hash)
        rows[r.episode_hash] = r

    monkeypatch.setattr(run_conversion, "add_episode", add)
    monkeypatch.setattr(
        run_conversion,
        "update_episode",
        lambda e, r: rows.__setitem__(r.episode_hash, r),
    )
    return inserted


REC_MS = 1794800772000


def _demo(offset_s, frames=500, rec_ms=REC_MS):
    h = timestamp_ms_to_episode_hash(rec_ms + int(offset_s * 1000))
    return {
        "episode_hash": h,
        "ds_path": f"/local/{h}.zarr",
        "mp4_path": f"/local/{h}.mp4",
        "frames": frames,
    }


def _recording(run_conversion, rec_ms=REC_MS, **kw):
    fields = dict(
        operator="Elmo",
        lab="rl2",
        task="fold_clothes",
        embodiment="aria_gen2",
        rig_name="aria_gen2",
        objects="{cloth}",
    )
    return run_conversion.TableRow(
        episode_hash=timestamp_ms_to_episode_hash(rec_ms), **{**fields, **kw}
    )


def test_record_demo_rows_inserts_children_and_retires_parent(monkeypatch):
    run_conversion = pytest.importorskip("egomimic.scripts.run_conversion")
    parent = _recording(run_conversion, is_eval=True)
    rows = {parent.episode_hash: parent}
    inserted = _fake_sql(monkeypatch, run_conversion, rows)
    demos = [_demo(4.5), _demo(56.0, frames=400)]
    run_conversion._record_demo_rows(None, parent, demos, PREFIX)
    assert inserted == [d["episode_hash"] for d in demos]
    child = rows[demos[0]["episode_hash"]]
    assert child.task == "fold_clothes" and child.is_eval  # eval stays eval
    assert child.zarr_processed_path == (
        f"{PREFIX}/{parent.episode_hash}/{demos[0]['episode_hash']}.zarr"
    )
    assert rows[parent.episode_hash].zarr_processing_error.startswith(
        "split into 2 demos"
    )
    assert rows[parent.episode_hash].num_frames == 900

    demos[1]["frames"] = -2  # an upload failure leaves the parent to be redone
    run_conversion._record_demo_rows(None, parent, demos, PREFIX)
    assert rows[demos[1]["episode_hash"]].zarr_processing_error.startswith(
        "Upload Failed [recording "
    )
    assert rows[parent.episode_hash].zarr_processing_error == ""


def test_redo_merges_retires_stale_and_deleted_counts_as_done(monkeypatch):
    run_conversion = pytest.importorskip("egomimic.scripts.run_conversion")
    parent = _recording(run_conversion)
    rows = {parent.episode_hash: parent}
    _fake_sql(monkeypatch, run_conversion, rows)
    first = [_demo(4.5), _demo(56.0), _demo(92.0)]
    run_conversion._record_demo_rows(None, parent, first, PREFIX)
    rows[first[0]["episode_hash"]].eval_score = 0.9  # set by someone later
    rows[first[1]["episode_hash"]].is_deleted = True  # deleted by a human

    redo = [_demo(4.5), _demo(56.0), _demo(80.0)]  # new settings: demo 3 moved
    run_conversion._record_demo_rows(None, parent, redo, PREFIX)
    assert rows[first[0]["episode_hash"]].eval_score == 0.9  # merged, kept
    assert rows[first[1]["episode_hash"]].is_deleted  # not resurrected
    assert rows[first[2]["episode_hash"]].is_deleted  # stale: retired
    assert not rows[redo[2]["episode_hash"]].is_deleted
    # the deleted demo counts as done, so the recording is not redone forever
    assert rows[parent.episode_hash].zarr_processing_error.startswith(
        "split into 3 demos"
    )
    assert not rows[parent.episode_hash].is_deleted


def test_failed_split_redo_keeps_existing_demos(monkeypatch):
    run_conversion = pytest.importorskip("egomimic.scripts.run_conversion")
    parent = _recording(run_conversion)
    rows = {parent.episode_hash: parent}
    _fake_sql(monkeypatch, run_conversion, rows)
    demos = [_demo(4.5), _demo(56.0)]
    run_conversion._record_demo_rows(None, parent, demos, PREFIX)

    # A split-mode redo that fails returns the ("", "", -1) failure tuple.
    for frames in (-1, -2, None):
        run_conversion._retire_demos_after_single_episode(
            None, parent, PREFIX, split_demos=True, frames=frames
        )
    # ... and so does a non-split run that fails.
    run_conversion._retire_demos_after_single_episode(
        None, parent, PREFIX, split_demos=False, frames=-1
    )
    assert not any(rows[d["episode_hash"]].is_deleted for d in demos)

    # Only a successful non-split conversion replaces the demos.
    run_conversion._retire_demos_after_single_episode(
        None, parent, PREFIX, split_demos=False, frames=1200
    )
    assert all(rows[d["episode_hash"]].is_deleted for d in demos)


def test_recording_frames_count_only_live_demos(monkeypatch):
    run_conversion = pytest.importorskip("egomimic.scripts.run_conversion")
    parent = _recording(run_conversion)
    rows = {parent.episode_hash: parent}
    _fake_sql(monkeypatch, run_conversion, rows)
    demos = [_demo(4.5, frames=500), _demo(56.0, frames=400)]
    run_conversion._record_demo_rows(None, parent, demos, PREFIX)
    rows[demos[1]["episode_hash"]].is_deleted = True
    run_conversion._record_demo_rows(None, parent, demos, PREFIX)
    assert rows[parent.episode_hash].num_frames == 500


def test_overlapping_same_rig_recordings_stay_apart(monkeypatch):
    run_conversion = pytest.importorskip("egomimic.scripts.run_conversion")
    a = _recording(run_conversion)
    b = _recording(run_conversion, rec_ms=REC_MS + 20_000, operator="Other")
    rows = {a.episode_hash: a, b.episode_hash: b}
    _fake_sql(monkeypatch, run_conversion, rows)
    run_conversion._record_demo_rows(
        None, b, [_demo(1.0, rec_ms=REC_MS + 20_000)], PREFIX
    )
    b_demo = _demo(1.0, rec_ms=REC_MS + 20_000)["episode_hash"]  # inside A's span

    # Redo A with demos that do not include B's: B's rows must be untouched.
    run_conversion._record_demo_rows(None, a, [_demo(4.5), _demo(56.0)], PREFIX)
    assert not rows[b_demo].is_deleted and not rows[b.episode_hash].is_deleted
    assert rows[b_demo].operator == "Other"

    # A demo of A landing on B's demo hash is a clash, not an overwrite.
    run_conversion._record_demo_rows(None, a, [_demo(21.0)], PREFIX)
    assert rows[b_demo].operator == "Other"
    assert rows[a.episode_hash].zarr_processing_error.startswith("Demo hash clash")
    assert not rows[a.episode_hash].is_deleted


def test_missing_gaze_is_recorded_as_unavailable_reset_detection():
    rgb = np.array([0, 33_000_000], dtype=np.int64)
    sig = aria_utils.gen2_frame_signals(
        rgb,
        [_pose(0), _pose(33_000_000)],
        [_hands(0), _hands(33_000_000)],
        [],
        np.ones(2, bool),
        CFG,
    )
    assert not sig.gaze_available and sig.gaze_valid.all()


def test_utc_timestamps_from_vrs_sync_and_fallback():
    device = np.array([1_000, 34_000_000, 67_000_000], dtype=np.int64)
    offset = 1794800772_123_456_789 - 1_000

    class Provider:
        def __init__(self, utc):
            self.utc = utc

        def supports_time_domain(self, stream, domain):
            return self.utc

        def get_timestamps_ns(self, stream, domain):
            return list(device + offset) if domain.name == "UTC" else list(device)

        def get_metadata(self):
            return SimpleNamespace(start_time_epoch_sec=1794800772)

    utc = aria_utils.gen2_utc_timestamps_ns(Provider(True), device[1:])
    np.testing.assert_array_equal(utc, device[1:] + offset)
    fallback = aria_utils.gen2_utc_timestamps_ns(Provider(False), device)
    assert fallback[0] == 1794800772_000_000_000


def test_aria_keypoints_reorder_to_canonical_layout():
    aria = np.arange(21 * 3, dtype=float).reshape(1, 21, 3)
    out = aria_utils.aria_keypoints_to_canonical_layout(aria.reshape(1, 63))
    out = out.reshape(21, 3)
    # canonical: 0 wrist <- aria 5, 5 index MCP <- aria 8, 9 middle MCP <- 11,
    # 13 ring MCP <- 14, 17 pinky MCP <- 17, 4 thumb tip <- aria 0
    for canonical, aria_idx in ((0, 5), (5, 8), (9, 11), (13, 14), (17, 17), (4, 0)):
        np.testing.assert_array_equal(out[canonical], aria[0, aria_idx])
    assert np.isnan(out[1]).all()  # thumb CMC has no Aria landmark
    sentinel = np.full((1, 63), 1e9)
    assert np.isnan(aria_utils.aria_keypoints_to_canonical_layout(sentinel)).all()
