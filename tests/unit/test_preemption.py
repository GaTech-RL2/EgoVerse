"""PreemptionCheckpoint: save a checkpoint on the Slurm preemption signal, then requeue.

Under ``hydra -m`` submitit owns the preemption signal (``USR2`` by default, or
``$SUBMITIT_PREEMPT_SIGNAL``) and requeues the job *without* saving a Lightning
checkpoint. Lightning refuses to register its own handler on a signal that already
has one, so the callback installs a handler that saves first and then chains to
the handler that was installed before it (submitit's requeue).
"""

import os
import signal

import pytest
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel

from egomimic.pl_utils.preemption import PreemptionCheckpoint

SIG = signal.SIGUSR2


class _RaiseSignal(Callback):
    """Deliver ``signum`` to this process from inside the training loop."""

    def __init__(self, signum: signal.Signals):
        self.signum = signum

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 1:
            signal.raise_signal(self.signum)


def _fit(tmp_path, *callbacks, max_steps=4, ckpt_path=None):
    trainer = Trainer(
        default_root_dir=str(tmp_path),
        max_steps=max_steps,
        limit_train_batches=max_steps,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=any(isinstance(c, ModelCheckpoint) for c in callbacks),
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[*callbacks, _RaiseSignal(SIG)],
    )
    trainer.fit(BoringModel(), ckpt_path=ckpt_path)
    return trainer


@pytest.fixture
def restore_sig():
    old = signal.getsignal(SIG)
    yield
    signal.signal(SIG, old)


def test_saves_checkpoint_then_chains_to_previous_handler(tmp_path, restore_sig):
    seen = []
    prev = lambda signum, frame: seen.append(signum)  # noqa: E731
    signal.signal(SIG, prev)
    ckpt = tmp_path / "ckpts" / "preempt.ckpt"

    _fit(tmp_path, PreemptionCheckpoint(ckpt_path=str(ckpt), signum=SIG))

    assert ckpt.exists(), "checkpoint was not written on the preemption signal"
    assert seen == [SIG], "previous (submitit) handler was not chained"
    assert signal.getsignal(SIG) is prev, "previous handler was not restored after fit"


def test_default_ckpt_path_is_last_ckpt_under_trainer_root(tmp_path, restore_sig):
    signal.signal(SIG, lambda s, f: None)

    _fit(tmp_path, PreemptionCheckpoint(signum=SIG))

    assert (tmp_path / "checkpoints" / "last.ckpt").exists()


def test_default_signal_follows_submitit_env(monkeypatch):
    monkeypatch.delenv("SUBMITIT_PREEMPT_SIGNAL", raising=False)
    assert PreemptionCheckpoint().signum == signal.SIGUSR2

    monkeypatch.setenv("SUBMITIT_PREEMPT_SIGNAL", "USR1")
    assert PreemptionCheckpoint().signum == signal.SIGUSR1


def test_save_failure_still_chains_to_requeue(tmp_path, restore_sig, monkeypatch):
    seen = []
    signal.signal(SIG, lambda signum, frame: seen.append(signum))

    def boom(self, *a, **k):
        raise OSError("disk quota exceeded")

    monkeypatch.setattr(Trainer, "save_checkpoint", boom)
    cb = PreemptionCheckpoint(ckpt_path=str(tmp_path / "last.ckpt"), signum=SIG)

    _fit(tmp_path, cb)  # the OSError must not escape the handler either

    assert seen == [SIG], "a failed save must not cancel the (submitit) requeue"


def test_no_temp_file_left_next_to_checkpoint(tmp_path, restore_sig):
    signal.signal(SIG, lambda s, f: None)
    ckpt = tmp_path / "last.ckpt"

    _fit(tmp_path, PreemptionCheckpoint(ckpt_path=str(ckpt), signum=SIG))

    assert ckpt.exists()
    assert not (tmp_path / "last.ckpt.preempt").exists()


def test_coexists_with_model_checkpoint_save_last(tmp_path, restore_sig):
    """Preempt before ModelCheckpoint's first periodic save (the real config saves
    every 100 epochs), then resume. ModelCheckpoint must keep writing last.ckpt
    rather than versioning to last-v1.ckpt, which trainHydra never resumes from."""
    signal.signal(SIG, lambda s, f: None)
    ckpt_dir = tmp_path / "checkpoints"

    # Same arguments in both fits, as in a real resume: Lightning only restores
    # ModelCheckpoint state into a callback with an identical state key.
    def mc():
        return ModelCheckpoint(
            dirpath=str(ckpt_dir), save_last=True, every_n_train_steps=4, save_top_k=0
        )

    # Signal fires at global_step 2, before the first periodic save at step 4.
    _fit(tmp_path, PreemptionCheckpoint(signum=SIG), mc(), max_steps=3)
    assert (ckpt_dir / "last.ckpt").exists()

    _fit(tmp_path, mc(), max_steps=8, ckpt_path=str(ckpt_dir / "last.ckpt"))

    files = sorted(p.name for p in ckpt_dir.glob("*.ckpt"))
    assert files == ["last.ckpt"], f"ModelCheckpoint versioned last.ckpt: {files}"
    step = torch.load(ckpt_dir / "last.ckpt", map_location="cpu", weights_only=False)[
        "global_step"
    ]
    assert step == 8, f"last.ckpt is stale (global_step={step})"


def test_coexists_with_model_checkpoint_through_symlinked_run_dir(
    tmp_path, restore_sig
):
    """Lightning realpath()s ModelCheckpoint.dirpath but only normpath()s
    default_root_dir; the claim must still match through a symlink."""
    signal.signal(SIG, lambda s, f: None)
    real = tmp_path / "real_run"
    real.mkdir()
    link = tmp_path / "link_run"
    link.symlink_to(real, target_is_directory=True)
    ckpt_dir = link / "checkpoints"

    def mc():
        return ModelCheckpoint(
            dirpath=str(ckpt_dir), save_last=True, every_n_train_steps=4, save_top_k=0
        )

    _fit(link, PreemptionCheckpoint(signum=SIG), mc(), max_steps=3)
    _fit(link, mc(), max_steps=8, ckpt_path=str(ckpt_dir / "last.ckpt"))

    files = sorted(p.name for p in (real / "checkpoints").glob("*.ckpt"))
    assert files == ["last.ckpt"], f"ModelCheckpoint versioned last.ckpt: {files}"


class _ForkAndSignal(Callback):
    """Deliver the signal in a forked child (a DataLoader worker stand-in)."""

    def __init__(self, signum):
        self.signum = signum

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx != 1:
            return
        pid = os.fork()
        if pid == 0:  # child: inherits the parent's handler table
            signal.raise_signal(self.signum)
            os._exit(0)
        _, status = os.waitpid(pid, 0)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def test_forked_worker_does_not_checkpoint(tmp_path, restore_sig):
    """Slurm signals every process in the step, including DataLoader workers,
    which inherit the handler. Only the process that installed it may act."""
    signal.signal(SIG, signal.SIG_IGN)
    ckpt = tmp_path / "last.ckpt"
    trainer = Trainer(
        default_root_dir=str(tmp_path),
        max_steps=4,
        limit_train_batches=4,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            PreemptionCheckpoint(ckpt_path=str(ckpt), signum=SIG),
            _ForkAndSignal(SIG),
        ],
    )
    trainer.fit(BoringModel())

    assert not ckpt.exists(), "a forked child wrote the preemption checkpoint"
