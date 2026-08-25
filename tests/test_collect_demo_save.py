from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from egomimic.robot import collect_demo


class _ZeroPoseSolver:
    def __init__(self, model_path):
        self.model_path = model_path

    def fk(self, joints):  # pragma: no cover - zero-joint fixtures skip FK
        raise AssertionError("FK should not run for zero-joint fixtures")


def _demo_data(steps: int, *, identical_images: bool = False) -> dict:
    observations = []
    robot_joint_actions = []
    cmd_joint_actions = []
    cmd_eepose_actions = []
    for index in range(steps):
        pixel_value = 7 if identical_images else index % 256
        observations.append({"camera": np.full((2, 3, 3), pixel_value, dtype=np.uint8)})
        robot_joint_actions.append(np.zeros(14, dtype=np.float32))
        cmd_joint_actions.append(np.full(14, index, dtype=np.float32))
        cmd_eepose_actions.append(np.full(14, index + 1, dtype=np.float32))
    return {
        "obs": observations,
        "robot_joint_actions": robot_joint_actions,
        "cmd_joint_actions": cmd_joint_actions,
        "cmd_eepose_actions": cmd_eepose_actions,
    }


@pytest.fixture(autouse=True)
def _fake_kinematics(monkeypatch):
    monkeypatch.setattr(collect_demo, "EvaMinkKinematicsSolver", _ZeroPoseSolver)


def _partials(directory: Path) -> list[Path]:
    return list(directory.glob(".demo_*.hdf5.partial"))


def test_save_demo_rejects_zero_steps_without_creating_a_file(tmp_path, capsys):
    assert (
        collect_demo.save_demo(_demo_data(0), tmp_path, 5, {"camera": (2, 3)}) is False
    )

    assert not (tmp_path / "demo_5.hdf5").exists()
    assert _partials(tmp_path) == []
    assert "zero steps" in capsys.readouterr().out


def test_save_demo_rejects_stuck_camera_without_creating_a_file(tmp_path):
    assert (
        collect_demo.save_demo(
            _demo_data(101, identical_images=True),
            tmp_path,
            6,
            {"camera": (2, 3)},
        )
        is False
    )

    assert not (tmp_path / "demo_6.hdf5").exists()
    assert _partials(tmp_path) == []


def test_save_demo_publishes_complete_file_without_mutating_input(tmp_path):
    demo_data = _demo_data(1)

    assert collect_demo.save_demo(demo_data, tmp_path, 7, {"camera": (2, 3)}) is True

    destination = tmp_path / "demo_7.hdf5"
    assert destination.exists()
    assert _partials(tmp_path) == []
    assert len(demo_data["obs"]) == 1
    with h5py.File(destination, "r") as recording:
        assert recording["action"].shape == (1, 14)
        assert recording["observations/images/camera"].shape == (1, 2, 3, 3)
        np.testing.assert_array_equal(
            recording["observations/images/camera"][0],
            demo_data["obs"][0]["camera"][..., ::-1],
        )


def test_save_demo_refuses_to_overwrite_existing_episode(tmp_path):
    destination = tmp_path / "demo_8.hdf5"
    destination.write_bytes(b"existing-demo")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        collect_demo.save_demo(_demo_data(1), tmp_path, 8, {"camera": (2, 3)})

    assert destination.read_bytes() == b"existing-demo"
    assert _partials(tmp_path) == []


def test_save_demo_publish_race_does_not_overwrite(monkeypatch, tmp_path):
    destination = tmp_path / "demo_9.hdf5"
    real_link = collect_demo.os.link

    def race_link(source, target):
        destination.write_bytes(b"raced-demo")
        return real_link(source, target)

    monkeypatch.setattr(collect_demo.os, "link", race_link)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        collect_demo.save_demo(_demo_data(1), tmp_path, 9, {"camera": (2, 3)})

    assert destination.read_bytes() == b"raced-demo"
    assert _partials(tmp_path) == []


def test_save_demo_write_failure_leaves_no_partial_or_final(monkeypatch, tmp_path):
    demo_data = _demo_data(1)

    def fail_to_sync(*args, **kwargs):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(collect_demo.os, "fsync", fail_to_sync)

    with pytest.raises(OSError, match="injected fsync failure"):
        collect_demo.save_demo(demo_data, tmp_path, 10, {"camera": (2, 3)})

    assert not (tmp_path / "demo_10.hdf5").exists()
    assert _partials(tmp_path) == []
    assert len(demo_data["obs"]) == 1


@pytest.mark.parametrize(
    ("saved", "current_id", "expected_id"),
    [(False, 11, 11), (True, 11, 12), (False, None, None), (True, None, None)],
)
def test_episode_id_advances_only_after_successful_save(
    monkeypatch, tmp_path, saved, current_id, expected_id
):
    monkeypatch.setattr(collect_demo, "save_demo", lambda *args, **kwargs: saved)

    assert (
        collect_demo._save_demo_and_advance_episode({}, tmp_path, 11, {}, current_id)
        == expected_id
    )
