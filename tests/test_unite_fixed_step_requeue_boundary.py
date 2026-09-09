import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from egomimic.utils.slurm_requeue import SaveOnlySignalCheckpoint
from scripts.ice.unite.fixed_step_requeue_boundary import FixedStepRequeueBoundary


class _Strategy:
    world_size = 2

    def __init__(self):
        self.barriers = []

    def barrier(self, name):
        self.barriers.append(name)

    @staticmethod
    def broadcast(value, src=0):
        return value


def _callback(tmp_path: Path, scancel: Path) -> FixedStepRequeueBoundary:
    return FixedStepRequeueBoundary(
        initial_global_step=3,
        trigger_global_step=8,
        slurm_job_id="1234",
        scancel=str(scancel),
        record_path=str(tmp_path / "boundary.json"),
        signal_ack_timeout_seconds=0.1,
        requeue_wait_timeout_seconds=0.01,
    )


def test_boundary_requires_exactly_five_steps(tmp_path):
    executable = tmp_path / "scancel"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    with pytest.raises(ValueError, match="exactly five"):
        FixedStepRequeueBoundary(
            initial_global_step=3,
            trigger_global_step=9,
            slurm_job_id="1234",
            scancel=str(executable),
            record_path=str(tmp_path / "boundary.json"),
        )


def test_step_eight_requests_boundary_and_records_all_rank_ack(
    monkeypatch, tmp_path
):
    executable = tmp_path / "scancel"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    callback = _callback(tmp_path, executable)
    save_callback = SaveOnlySignalCheckpoint(tmp_path / "checkpoints")
    strategy = _Strategy()
    trainer = SimpleNamespace(
        callbacks=[callback, save_callback],
        global_step=8,
        is_global_zero=True,
        strategy=strategy,
    )
    callback.setup(trainer, SimpleNamespace(), "fit")
    monkeypatch.setenv("ICE_REQUEUE_OWNER", "runner")
    monkeypatch.setenv("ICE_CHILD_REQUEUE_DISABLED", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)

    def request(*_args, **_kwargs):
        save_callback._request_generation += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", request)
    callback.on_train_batch_end(trainer, SimpleNamespace(), None, None, 0)

    proof = json.loads((tmp_path / "boundary.json").read_text(encoding="utf-8"))
    assert proof["observed_global_step"] == 8
    assert proof["optimizer_steps_advanced"] == 5
    assert proof["step_after_signal_executed"] is False
    assert proof["save_acknowledged_ranks"] == [0, 1]
    assert proof["world_size"] == 2
    assert strategy.barriers == [
        "unite-fixed-step-boundary-ready",
        "unite-fixed-step-boundary-signal-acknowledged",
    ]


def test_boundary_rejects_wrong_owner_before_signalling(monkeypatch, tmp_path):
    executable = tmp_path / "scancel"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    callback = _callback(tmp_path, executable)
    save_callback = SaveOnlySignalCheckpoint(tmp_path / "checkpoints")
    trainer = SimpleNamespace(
        callbacks=[callback, save_callback],
        global_step=8,
        is_global_zero=True,
        strategy=_Strategy(),
    )
    callback.setup(trainer, SimpleNamespace(), "fit")
    monkeypatch.setenv("ICE_REQUEUE_OWNER", "child")
    monkeypatch.setenv("ICE_CHILD_REQUEUE_DISABLED", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")

    with pytest.raises(RuntimeError, match="sole requeue owner"):
        callback.on_train_batch_end(trainer, SimpleNamespace(), None, None, 0)
    assert not (tmp_path / "boundary.json").exists()


def test_step_after_boundary_is_blocked(monkeypatch, tmp_path):
    executable = tmp_path / "scancel"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    callback = _callback(tmp_path, executable)
    callback._sent = True
    monotonic_values = iter((0.0, 0.0, 0.02))
    monkeypatch.setattr("time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="did not terminate"):
        callback.on_train_batch_start(SimpleNamespace(), SimpleNamespace(), None, 0)
