import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from egomimic.utils.slurm_requeue import (
    SaveOnlySignalCheckpoint,
    _exclusive_reservation,
    _publish_hardlink_no_overwrite,
)


class _GlooStrategy:
    @property
    def world_size(self):
        return dist.get_world_size()

    def broadcast(self, value, src=0):
        values = [value]
        dist.broadcast_object_list(values, src=src)
        return values[0]


class _FakeDistributedTrainer:
    def __init__(self, checkpoint_dir, rank):
        self.current_epoch = 7
        self.global_step = 1234
        self.is_global_zero = rank == 0
        self.strategy = _GlooStrategy()
        self._checkpoint_dir = Path(checkpoint_dir)
        self._rank = rank

    def save_checkpoint(self, path, weights_only=None):
        Path(self._checkpoint_dir / f"rank-{self._rank}-save-call.json").write_text(
            json.dumps({"path": path, "weights_only": weights_only})
        )
        if self.is_global_zero:
            temporary = Path(f"{path}.tmp")
            torch.save(
                {
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                    "state_dict": {"weight": torch.tensor([1.0])},
                    "optimizer_states": [{"state": {}, "param_groups": []}],
                    "lr_schedulers": [{"_last_lr": [3.0e-5]}],
                    "loops": {"fit_loop": {"state_dict": {}}},
                    "ema_state_dict": {"weight": torch.tensor([1.0])},
                },
                temporary,
            )
            os.replace(temporary, path)
        dist.barrier()


def _gloo_save_worker(rank, world_size, init_file, checkpoint_dir):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        callback = SaveOnlySignalCheckpoint(checkpoint_dir)
        callback._request_checkpoint(callback.save_signal.value, None)
        trainer = _FakeDistributedTrainer(checkpoint_dir, rank)
        callback.on_validation_batch_end(
            trainer,
            SimpleNamespace(),
            None,
            None,
            0,
            0,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_sigusr2_request_saves_full_state_on_both_ranks_during_validation(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SLURM_JOB_ID", "9876")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "3")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    init_file = tmp_path / "gloo-init"

    mp.spawn(
        _gloo_save_worker,
        args=(2, str(init_file), str(checkpoint_dir)),
        nprocs=2,
        join=True,
    )

    expected = checkpoint_dir / (
        "signal-job=9876-restart=003-request=000-epoch=7-step=1234.ckpt"
    )
    proof_path = expected.with_suffix(expected.suffix + ".save-only.json")
    partial_path = expected.with_name(f".{expected.name}.partial")
    reservation_path = expected.with_name(f".{expected.name}.reserve")
    assert expected.is_file()
    assert proof_path.is_file()
    assert not partial_path.exists()
    assert not reservation_path.exists()
    for rank in range(2):
        call = json.loads((checkpoint_dir / f"rank-{rank}-save-call.json").read_text())
        assert call == {"path": str(partial_path), "weights_only": False}

    checkpoint = torch.load(expected, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 1234
    assert checkpoint["optimizer_states"]
    assert checkpoint["lr_schedulers"]
    assert checkpoint["loops"]
    assert checkpoint["ema_state_dict"]

    proof = json.loads(proof_path.read_text())
    digest = hashlib.sha256(expected.read_bytes()).hexdigest()
    assert proof["status"] == "FULL_STATE_CHECKPOINT_PUBLISHED_UNVALIDATED"
    assert proof["checkpoint_path"] == str(expected)
    assert proof["checkpoint_sha256"] == digest
    assert proof["global_step"] == 1234
    assert proof["restart_count"] == 3
    assert proof["save_signal"] == "SIGUSR2"
    assert proof["hook"] == "on_validation_batch_end"
    assert proof["world_size"] == 2
    assert proof["all_ranks_save_checkpoint_returned"] is True
    assert proof["weights_only"] is False
    assert proof["publish_atomic_no_overwrite"] is True
    assert proof["partial_path"] == str(partial_path)
    assert proof["partial_path_absent"] is True
    assert proof["reservation_path"] == str(reservation_path)
    assert proof["reservation_path_absent"] is True
    assert proof["application_validation_status"] == "NOT_RUN_BY_CALLBACK"
    assert proof["scontrol_invoked"] is False
    assert proof["wandb_finalize_invoked"] is False


def test_uneven_no_signal_hook_counts_do_no_collective_or_save(tmp_path):
    class _NoCollectiveStrategy:
        world_size = 2

        @staticmethod
        def broadcast(*_args, **_kwargs):
            raise AssertionError("no-signal hook entered a collective")

    class _NoSaveTrainer:
        current_epoch = 7
        global_step = 1234
        strategy = _NoCollectiveStrategy()
        is_global_zero = True

        @staticmethod
        def save_checkpoint(*_args, **_kwargs):
            raise AssertionError("no-signal hook attempted a checkpoint")

    rank_zero_callback = SaveOnlySignalCheckpoint(tmp_path)
    rank_one_callback = SaveOnlySignalCheckpoint(tmp_path)
    trainer = _NoSaveTrainer()
    for batch_index in range(3):
        rank_zero_callback.on_train_batch_end(
            trainer,
            SimpleNamespace(),
            None,
            None,
            batch_index,
        )
    for batch_index in range(5):
        rank_one_callback.on_validation_batch_end(
            trainer,
            SimpleNamespace(),
            None,
            None,
            batch_index,
            0,
        )

    assert list(tmp_path.iterdir()) == []


def test_concurrent_reservation_allows_exactly_one_owner(tmp_path):
    reservation = tmp_path / ".checkpoint.reserve"
    barrier = threading.Barrier(2)

    def attempt(owner):
        barrier.wait()
        try:
            _exclusive_reservation(reservation, {"owner": owner})
        except FileExistsError:
            return ("collision", owner)
        return ("reserved", owner)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, (0, 1)))

    assert [status for status, _owner in results].count("reserved") == 1
    assert [status for status, _owner in results].count("collision") == 1
    winner = next(owner for status, owner in results if status == "reserved")
    assert json.loads(reservation.read_text()) == {"owner": winner}


def test_concurrent_hardlink_publish_never_overwrites_winner(tmp_path):
    partials = [tmp_path / ".a.partial", tmp_path / ".b.partial"]
    partials[0].write_bytes(b"candidate-a")
    partials[1].write_bytes(b"candidate-b")
    final = tmp_path / "checkpoint.ckpt"
    barrier = threading.Barrier(2)

    def attempt(index):
        barrier.wait()
        try:
            _publish_hardlink_no_overwrite(partials[index], final)
        except FileExistsError:
            return ("collision", index)
        return ("published", index)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, (0, 1)))

    assert [status for status, _index in results].count("published") == 1
    assert [status for status, _index in results].count("collision") == 1
    winner = next(index for status, index in results if status == "published")
    assert final.read_bytes() == partials[winner].read_bytes()
    winning_bytes = final.read_bytes()
    with pytest.raises(FileExistsError):
        _publish_hardlink_no_overwrite(partials[1 - winner], final)
    assert final.read_bytes() == winning_bytes


def test_same_attempt_same_step_never_overwrites_existing_checkpoint(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SLURM_JOB_ID", "9876")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "3")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    existing = checkpoint_dir / (
        "signal-job=9876-restart=003-request=000-epoch=7-step=1234.ckpt"
    )
    existing.write_bytes(b"immutable")

    class _SingleStrategy:
        world_size = 1

        @staticmethod
        def broadcast(value, src=0):
            return value

    trainer = SimpleNamespace(
        current_epoch=7,
        global_step=1234,
        strategy=_SingleStrategy(),
        is_global_zero=True,
    )
    callback = SaveOnlySignalCheckpoint(checkpoint_dir)
    callback._request_checkpoint(callback.save_signal.value, None)

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        callback._service(trainer, "test")

    assert existing.read_bytes() == b"immutable"
    assert not existing.with_name(f".{existing.name}.partial").exists()
    assert not existing.with_name(f".{existing.name}.reserve").exists()
