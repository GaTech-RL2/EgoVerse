"""Aria Gen 2 calibration / reset filter and demo segmentation."""

import numpy as np
import pytest

from egomimic.scripts.aria_process.gen2_filter import (
    REASONS,
    Gen2FilterConfig,
    Gen2FrameSignals,
    classify_frames,
    demo_segments,
    first_reason,
)

FPS = 30
CFG = Gen2FilterConfig()


def _signals(seconds):
    T = int(seconds * FPS)
    return Gen2FrameSignals(
        t_ns=(np.arange(T) * 1e9 / FPS).astype(np.int64),
        tracking_ok=np.ones(T, bool),
        gaze_valid=np.ones(T, bool),
        # Looking down at the table, hands 0.5 m away.
        device_forward_world=np.tile([0.0, 0.0, -1.0], (T, 1)),
        hand_min_dist_m=np.full(T, 0.5),
        left_tracked=np.ones(T, bool),
        right_tracked=np.ones(T, bool),
    )


def _sec(sig, a, b):
    t = sig.t_ns / 1e9
    return (t >= a) & (t < b)


def _dropped(masks):
    return masks["boundary"] | masks["error"]


def test_clean_task_keeps_everything():
    masks = classify_frames(_signals(20), "both", CFG)
    assert not _dropped(masks).any()


def test_reset_is_a_boundary_with_padding():
    sig = _signals(30)
    sig.gaze_valid[_sec(sig, 10, 15)] = False  # glasses on the forehead
    masks = classify_frames(sig, "both", CFG)
    assert masks["boundary"][_sec(sig, 9.1, 15.9)].all()
    assert not masks["boundary"][_sec(sig, 0, 8.9)].any()
    assert not masks["boundary"][_sec(sig, 16.1, 30)].any()
    assert not masks["error"].any()


def test_blink_is_kept():
    sig = _signals(20)
    sig.gaze_valid[_sec(sig, 5, 5.15)] = False
    assert not _dropped(classify_frames(sig, "both", CFG)).any()


def test_short_gaze_dropout_is_an_error_not_a_boundary():
    sig = _signals(30)
    sig.gaze_valid[_sec(sig, 10, 11.2)] = False  # 1.2 s: looked away, not a reset
    masks = classify_frames(sig, "both", CFG)
    assert not masks["boundary"].any()
    assert masks["gaze_dropout"][_sec(sig, 10, 11.2)].all()
    assert masks["error"][_sec(sig, 10, 11.2)].all()


def _nod(sig, a, b):
    t = sig.t_ns / 1e9
    m = _sec(sig, a, b)
    ang = np.radians(-50 + 40 * np.sin(2 * np.pi * t[m]))
    sig.device_forward_world[m] = np.stack(
        [np.zeros(m.sum()), np.cos(ang), np.sin(ang)], 1
    )


def test_late_glance_is_not_calibration():
    sig = _signals(40)
    _nod(sig, 25, 25.6)  # one quick glance mid-demo, no opening calibration
    masks = classify_frames(sig, "both", CFG)
    assert not masks["calibration"].any()
    assert not masks["boundary"][_sec(sig, 0, 24)].any()
    assert masks["head_sweep"][_sec(sig, 25, 25.6)].any()


def test_glance_long_after_calibration_does_not_extend_it():
    sig = _signals(40)
    _nod(sig, 0, 4)
    _nod(sig, 22, 22.6)  # > calib_max_gap_s after the calibration ends
    masks = classify_frames(sig, "both", CFG)
    assert masks["calibration"][_sec(sig, 0, 4)].all()
    assert not masks["calibration"][_sec(sig, 6, 40)].any()


def _look_and_back(sig, at_s, deg=45.0, dur_s=0.5):
    """One quick look away and back, centred at at_s."""
    t = sig.t_ns / 1e9
    m = _sec(sig, at_s - dur_s / 2, at_s + dur_s / 2)
    # from the baseline (looking straight down, -90 deg) and back to it
    ang = np.radians(-90 + deg * np.sin(np.pi * (t[m] - t[m][0]) / dur_s))
    sig.device_forward_world[m] = np.stack(
        [np.zeros(m.sum()), np.cos(ang), np.sin(ang)], 1
    )


def test_single_look_and_back_is_not_calibration():
    sig = _signals(30)
    _look_and_back(sig, 4.0)
    masks = classify_frames(sig, "both", CFG)
    assert masks["head_sweep"][_sec(sig, 3.9, 4.1)].all()
    assert not masks["calibration"].any()
    assert not masks["boundary"].any()


def test_glances_every_few_seconds_are_not_calibration():
    sig = _signals(40)
    for at in range(2, 30, 4):
        _look_and_back(sig, at)
    masks = classify_frames(sig, "both", CFG)
    assert not masks["calibration"].any()


def test_calibration_after_a_stray_early_glance_is_found():
    sig = _signals(60)
    _look_and_back(sig, 1.6)  # stray glance, then a 9 s pause
    _nod(sig, 11, 17)  # the real calibration
    _nod(sig, 20, 24)
    masks = classify_frames(sig, "both", CFG)
    assert masks["calibration"][_sec(sig, 0, 24)].all()
    assert not masks["calibration"][_sec(sig, 25, 60)].any()


def test_late_head_turn_without_calibration_is_not_calibration():
    sig = _signals(60)
    _nod(sig, 12, 13.6)  # one 1.6 s head turn at 12 s, nothing before it
    masks = classify_frames(sig, "both", CFG)
    assert not masks["calibration"].any()


def test_no_restart_after_a_partial_calibration():
    sig = _signals(60)
    _nod(sig, 2, 3)  # 1 s of sweeping: more than a glance, less than calibration
    _nod(sig, 11, 17)  # would qualify, but only a lone glance may be skipped
    masks = classify_frames(sig, "both", CFG)
    assert not masks["calibration"].any()


def test_opening_calibration_is_a_boundary_including_pauses():
    sig = _signals(40)
    t = sig.t_ns / 1e9
    # 0-4 s and 8-12 s: nodding at ~1 Hz, +-40 deg; 4-8 s: a pause.
    for a, b in ((0, 4), (8, 12)):
        m = _sec(sig, a, b)
        ang = np.radians(-50 + 40 * np.sin(2 * np.pi * t[m]))
        sig.device_forward_world[m] = np.stack(
            [np.zeros(m.sum()), np.cos(ang), np.sin(ang)], 1
        )
    masks = classify_frames(sig, "both", CFG)
    assert masks["boundary"][_sec(sig, 0, 12)].all()
    assert masks["calibration"][_sec(sig, 4, 8)].all()
    assert not _dropped(masks)[_sec(sig, 13, 40)].any()


def test_hand_at_headset_is_an_error_not_a_boundary():
    sig = _signals(20)
    sig.hand_min_dist_m[_sec(sig, 10, 11)] = 0.05
    masks = classify_frames(sig, "both", CFG)
    assert masks["error"][_sec(sig, 9.6, 11.4)].all()
    assert not masks["error"][_sec(sig, 0, 9.4)].any()
    assert not masks["boundary"].any()


def test_untracked_hand_counts_only_for_the_arms_used():
    sig = _signals(10)
    sig.left_tracked[_sec(sig, 2, 3)] = False
    assert classify_frames(sig, "both", CFG)["error"][_sec(sig, 2, 3)].all()
    assert classify_frames(sig, "left", CFG)["error"][_sec(sig, 2, 3)].all()
    assert not classify_frames(sig, "right", CFG)["error"].any()


def test_first_reason_prefers_the_most_specific_rule():
    sig = _signals(20)
    sig.tracking_ok[_sec(sig, 2, 3)] = False
    sig.gaze_valid[_sec(sig, 10, 15)] = False
    sig.right_tracked[_sec(sig, 10, 15)] = False
    label = first_reason(classify_frames(sig, "both", CFG))
    assert (label[_sec(sig, 2, 3)] == REASONS.index("not_tracking")).all()
    assert (label[_sec(sig, 11, 14)] == REASONS.index("reset")).all()
    assert (label[_sec(sig, 17, 20)] == -1).all()


def test_demo_segments_split_at_boundaries_and_drop_short_pieces():
    sig = _signals(60)
    boundary = np.zeros(len(sig.t_ns), bool)
    boundary[_sec(sig, 0, 5)] = True  # calibration
    boundary[_sec(sig, 25, 30)] = True  # reset
    boundary[_sec(sig, 33, 38)] = True  # reset; 30-33 s is a 3 s leftover
    demo, demos, removed = demo_segments(sig.t_ns, boundary, CFG.min_demo_s)
    assert len(demos) == 2
    assert (demo[_sec(sig, 5.1, 24.9)] == 0).all()
    assert (demo[_sec(sig, 38.1, 60)] == 1).all()
    assert (demo[_sec(sig, 30.1, 32.9)] == -1).all()
    assert len(removed) == 1
    assert (demo[boundary] == -1).all()


def test_config_from_yaml(tmp_path):
    path = tmp_path / "gen2.yaml"
    path.write_text("min_demo_s: 8.0\nreset_pad_s: 0.5\n")
    cfg = Gen2FilterConfig.from_yaml(path)
    assert cfg.min_demo_s == 8.0 and cfg.reset_pad_s == 0.5
    assert cfg.head_sweep_deg_s == CFG.head_sweep_deg_s
    path.write_text("min_demo: 8.0\n")
    with pytest.raises(ValueError):
        Gen2FilterConfig.from_yaml(path)
    for bad in ("min_demo_s: '8'\n", "min_demo_s: true\n", "min_demo_s: [8]\n"):
        path.write_text(bad)
        with pytest.raises(ValueError, match="numbers"):
            Gen2FilterConfig.from_yaml(path)


def test_mps_eye_gaze_falls_back_to_on_device():
    aria_utils = pytest.importorskip("egomimic.scripts.aria_process.aria_utils")
    mps_gaze, on_device = ["m"], ["d"]
    assert aria_utils.select_gen2_eye_gaze(mps_gaze, on_device) == (["m"], "mps")
    assert aria_utils.select_gen2_eye_gaze([], on_device) == (["d"], "on_device")
    assert aria_utils.select_gen2_eye_gaze([], []) == ([], "none")
