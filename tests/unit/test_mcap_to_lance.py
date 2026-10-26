"""MCAP -> Lance converter (egomimic/scripts/mcap_process) on tiny synthetic episodes."""

import shutil
import subprocess

import numpy as np
import pytest
from fixtures.mcap_episodes import T0, write_mcap_episode

from egomimic.rldb.lance import commit
from egomimic.rldb.mcap import spec

lance = pytest.importorskip("lance")


def _ffmpeg_has(codec: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True)
    return codec in out.stdout.decode()


if not _ffmpeg_has("libx264"):
    pytest.skip("needs ffmpeg with libx264", allow_module_level=True)

from egomimic.scripts.lance_commit import commit_staged  # noqa: E402
from egomimic.scripts.mcap_process.mcap_to_lance import (  # noqa: E402
    convert_and_stage,
)


def commit_all(tmp_path, replace=False):
    """Run the committer over every lab with staged work, annotations last."""
    staging = tmp_path / "staging"
    labs = sorted(p.name for p in staging.iterdir()) if staging.exists() else []
    for lab in sorted(labs, key=lambda x: x == "annotations"):
        commit_staged(
            str(staging),
            lab,
            str(tmp_path / "lance"),
            tmp_path / "work",
            replace=replace,
        )


def land(tmp_path, ep, replace=False, convert=None):
    r = (convert or convert_and_stage)(
        ep,
        tmp_path / "staging",
        tmp_path / "lance",
        vendor="acme",
        codec="libx264",
        mano=False,
        replace=replace,
    )
    commit_all(tmp_path, replace)
    return r


def run(tmp_path, embodiment="human_bimanual", episode_id="0", **kw):
    ep = write_mcap_episode(tmp_path / "eps", embodiment, episode_id, **kw)
    return land(tmp_path, ep)


def table(tmp_path, name):
    return lance.dataset(tmp_path / "lance" / name).to_table().to_pylist()


def steps_rows(tmp_path, embodiment="*"):
    """Rows of the one steps table (steps/<lab>/<embodiment>/<end_effector>.lance)."""
    (path,) = (tmp_path / "lance" / "steps").glob(f"*/{embodiment}/*.lance")
    return table(tmp_path, path.relative_to(tmp_path / "lance"))


def test_episode_lands_in_every_table(tmp_path):
    r = run(tmp_path)
    assert r.status == "staged" and r.problems == []
    assert r.recording_id == spec.recording_id("rl2", "acme-rec-0")
    assert (r.num_frames, r.tasks, r.annotations_staged) == (60, 2, 2)
    assert r.valid_frac == {"head": 1.0, "left": 1.0, "right": 1.0}

    (rec,) = table(tmp_path, "recordings.lance")
    assert (rec["lab"], rec["end_effector"], rec["license"]) == (
        "rl2",
        "hand",
        "CC-BY-4.0",
    )
    assert rec["operator"] == "op_7" and rec["num_frames"] == 60
    assert int(rec["captured_at"].timestamp() * 1000) == T0 // 10**6
    assert {k for k, _ in rec["video_path"]} == {"front_1", "front_1_480"}

    tasks = sorted(table(tmp_path, "tasks.lance"), key=lambda t: t["t_start_ns"])
    assert [t["task_name"] for t in tasks] == ["fold_clothes", "pack_clothes"]
    assert tasks[0]["t_start_ns"] == T0 and tasks[1]["objects"] == ["shirt", "basket"]
    labels = table(tmp_path, "annotations.lance")
    assert {a["text"] for a in labels} == {"pick up the shirt", "drop it in the basket"}
    # annotation_source when given, else the episode's lab.
    assert {a["annotation_vendor"] for a in labels} == {"acme", "rl2"}
    point = next(a for a in labels if a["text"] == "drop it in the basket")
    assert 0 < point["t_end_ns"] - point["t_start_ns"] <= 34_000_000  # one frame wide
    (alias,) = table(tmp_path, "aliases.lance")
    assert alias["recording_id"] == r.recording_id
    # Half-open: the alias covers the last frame.
    assert alias["t_end_ns"] > max(s["timestamp_ns"] for s in steps_rows(tmp_path))
    assert commit.list_files(tmp_path / "staging" / "mcap") == {}

    steps = steps_rows(tmp_path)
    ts = np.array([s["timestamp_ns"] for s in steps])
    assert ts[0] == T0 and (np.diff(ts) > 0).all()  # absolute Unix ns
    # Head pose = world->head (x drifts 1 mm/frame, z 1.5) @ head->camera (z + 0.1).
    np.testing.assert_allclose(
        steps[5]["head_pose"], [0.005, 0, 1.6, 1, 0, 0, 0], atol=1e-5
    )
    # Canonical MANO joints pass through, in the world frame.
    assert steps[5]["left"]["keypoints"][:3] == pytest.approx(
        [0.405, 0.2, 1.2], abs=1e-5
    )

    from torchcodec.decoders import VideoDecoder

    for cam, rel in rec["video_path"]:
        meta = VideoDecoder(str(tmp_path / "lance" / rel), seek_mode="exact").metadata
        assert meta.num_frames == 60, cam


def test_rerun_skips_and_replace_keeps_one_copy(tmp_path):
    assert run(tmp_path).status == "staged"
    assert run(tmp_path).status == "skipped"
    r = land(tmp_path, tmp_path / "eps" / "episode_0", replace=True)
    assert r.status == "staged"
    assert len(table(tmp_path, "recordings.lance")) == 1
    assert len(table(tmp_path, "tasks.lance")) == 2
    assert len(table(tmp_path, "annotations.lance")) == 2
    assert len(steps_rows(tmp_path)) == 60


def test_replace_drops_the_vendors_removed_labels(tmp_path):
    run(tmp_path)
    write_mcap_episode(tmp_path / "eps", labels_in=None)  # re-export without labels
    r = land(tmp_path, tmp_path / "eps" / "episode_0", replace=True)
    assert r.status == "staged"
    # A delivery is authoritative for its lab's labels and the sources it names: the
    # lab's label goes, the one acme wrote stays (this delivery names no acme labels).
    assert [a["annotation_vendor"] for a in table(tmp_path, "annotations.lance")] == [
        "acme"
    ]
    # Without /task the episode info is one task over the whole recording.
    (task,) = table(tmp_path, "tasks.lance")
    assert task["task_name"] == "fold_clothes" and task["t_start_ns"] == T0


def test_dropped_frame_becomes_an_invalid_filler_row(tmp_path):
    r = run(tmp_path, drop_video_frame=11)
    assert (r.num_frames, r.filler_frames) == (60, 1)
    steps = steps_rows(tmp_path)
    filler = steps[11]
    assert not (
        filler["head_pose_valid"] or filler["left_valid"] or filler["right_valid"]
    )
    assert filler["left"]["confidence"] == -1.0
    assert steps[12]["head_pose_valid"] is False  # first frame after the gap
    assert steps[13]["head_pose_valid"] is True


def test_single_arm_and_wrist_only_hands(tmp_path):
    r = run(tmp_path, "human_right_arm")
    assert r.valid_frac == {"head": 1.0, "left": 0.0, "right": 1.0}
    r = run(tmp_path, episode_id="w", hands_wrist_only=True)
    assert r.valid_frac["left"] == 1.0
    steps = [
        s
        for s in steps_rows(tmp_path, "human_bimanual")
        if s["recording_id"] == r.recording_id
    ]
    assert steps[0]["left"]["ee_pose"] == steps[0]["left"]["wrist_pose"]


def test_upright_rotation_turns_the_head_pose(tmp_path):
    run(tmp_path, upright_rotation_cw_deg=90)
    (s,) = [x for x in steps_rows(tmp_path)][5:6]
    # Rz(-90 deg) as wxyz, as aria_utils.ROTATION_MATRIX for Aria's sideways sensor.
    np.testing.assert_allclose(
        np.abs(s["head_pose"][3:]), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-5
    )


def test_failing_episode_is_rejected_and_nothing_committed(tmp_path):
    r = run(tmp_path, imu_hz=100)
    assert r.status == "rejected"
    assert r.problems == ["[imu] /imu: 100 Hz, need >= 200"]
    assert not (tmp_path / "lance" / "recordings.lance").exists()


def test_robot_episodes_wait_for_a_steps_schema(tmp_path):
    r = run(tmp_path, "eva_bimanual")
    assert r.status == "failed" and "no robot steps schema" in r.error


def test_hands_need_a_usable_head_pose(tmp_path):
    # Hands given in the head frame are placed with the head pose: lost head -> invalid.
    r = run(tmp_path, hand_frame="head", head_lost_from_s=1.0)
    steps = steps_rows(tmp_path)
    late = [s for s in steps if s["timestamp_ns"] >= T0 + 1_100_000_000]
    assert late and not any(s["head_pose_valid"] or s["left_valid"] for s in late)
    assert r.valid_frac["left"] == pytest.approx(r.valid_frac["head"], abs=0.02)
    # Hands given in the world frame do not depend on it.
    r = run(tmp_path, episode_id="w", head_lost_from_s=1.0)
    assert r.valid_frac["left"] == 1.0 and r.valid_frac["head"] < 0.6


def test_gaze_beyond_the_match_window_is_invalid(tmp_path):
    run(tmp_path, gaze_until_s=1.5)
    steps = steps_rows(tmp_path)
    late = [s for s in steps if s["timestamp_ns"] > T0 + 1_600_000_000]
    early = [s for s in steps if s["timestamp_ns"] < T0 + 1_400_000_000]
    assert late and all(s["eye_gaze"] == [0, 0, 0] for s in late)
    assert all(s["eye_gaze"] != [0, 0, 0] for s in early)


def test_packed_frame_past_the_validator_sample_fails(tmp_path):
    # 120 messages at 60 fps; the validator samples the first 60, frames pack from 100.
    r = run(tmp_path, video_fps=60, multi_frame_video=True, multi_frame_from=100)
    assert r.problems == []
    assert r.status == "failed" and "holds 2 frames, not 1" in r.error
    assert not (tmp_path / "lance" / "recordings.lance").exists()


def test_cli_keeps_going_past_a_failed_episode(tmp_path, capsys):
    from egomimic.scripts.mcap_process.mcap_to_lance import main

    robot = write_mcap_episode(tmp_path / "eps", "eva_bimanual", "robot")
    human = write_mcap_episode(tmp_path / "eps", "human_bimanual", "human")
    args = [
        "--stage-to",
        str(tmp_path / "staging"),
        "--lance",
        str(tmp_path / "lance"),
        "--vendor",
        "acme",
        "--codec",
        "libx264",
        "--no-mano",
    ]
    assert main([str(robot), str(human), *args]) == 1
    out = capsys.readouterr().out
    assert '"status": "failed"' in out and '"status": "staged"' in out
    assert not (tmp_path / "lance" / "recordings.lance").exists()
    commit_all(tmp_path)
    assert len(table(tmp_path, "recordings.lance")) == 1


def test_every_camera_becomes_a_video_with_its_calibration(tmp_path):
    r = run(tmp_path, slam_camera=True)
    assert r.videos == ["front_1", "front_1_480", "slam_left"]
    (rec,) = table(tmp_path, "recordings.lance")
    intr, extr = dict(rec["intrinsics"]), dict(rec["extrinsics"])
    # Native size and focal (64x48, f = 50), centred.
    assert intr["slam_left"] == pytest.approx([50, 0, 31.5, 0, 50, 23.5, 0, 0, 1])
    # Both cameras hang off the head at the same offset: identity relative to front_1.
    np.testing.assert_allclose(
        np.reshape(extr["slam_left"], (4, 4)), np.eye(4), atol=1e-9
    )
    from torchcodec.decoders import VideoDecoder

    rel = dict(rec["video_path"])["slam_left"]
    meta = VideoDecoder(str(tmp_path / "lance" / rel), seek_mode="exact").metadata
    assert (meta.num_frames, meta.height, meta.width) == (60, 48, 64)


def test_non_aria_sources_keep_their_field_of_view(tmp_path):
    run(tmp_path, metadata_override={"lab": "mecka"})
    (rec,) = table(tmp_path, "recordings.lance")
    # f = 50 px at 64 px wide -> 500 px at 640 wide; Aria would use 266.5.
    assert dict(rec["intrinsics"])["front_1_480"][0] == pytest.approx(500)
    assert len(steps_rows(tmp_path)) == 60


def test_labels_lance_would_refuse_reject_before_any_commit(tmp_path, monkeypatch):
    import egomimic.scripts.mcap_process.mcap_to_lance as m
    from egomimic.rldb.mcap.validate import Result

    def late_point(topic, msg, t):  # 10 ms before task 1 ends: Lance makes it 1 frame
        if topic == "/subtask-annotation" and not msg.end_time_unix_ns:
            return msg, T0 + 10**9 - 10_000_000

    ep = write_mcap_episode(tmp_path / "eps", tamper=late_point)
    # Even if a validator let it through, the converter checks before committing.
    monkeypatch.setattr(
        m, "validate_episode", lambda e, **kw: Result("human_bimanual", "x", [])
    )
    r = land(tmp_path, ep, convert=m.convert_and_stage)
    assert r.status == "rejected" and "not inside any task" in r.problems[0]
    assert not (tmp_path / "staging").exists()


def test_labels_delivered_after_the_commit_are_picked_up(tmp_path):
    r = run(tmp_path, labels_in=None)
    assert (r.status, r.annotations_staged) == ("staged", 0)
    write_mcap_episode(tmp_path / "eps", labels_without_tasks=True)  # labels arrive
    r = land(tmp_path, tmp_path / "eps" / "episode_0")
    assert (r.status, r.annotations_staged) == ("skipped", 2)
    assert len(table(tmp_path, "annotations.lance")) == 2
    (row,) = [
        c
        for c in table(tmp_path, "changes.lance")
        if c["source"] == "annotations" and c["added"]
    ]
    assert row["operator"] == "acme delivery"
