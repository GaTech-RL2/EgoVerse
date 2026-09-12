"""Checkpoint-on-preemption for Slurm runs.

Why this exists
---------------
Slurm delivers a warning signal before it preempts or times out a job. Two
things already listen for it:

* **submitit** (``hydra -m`` launches) installs a handler on ``USR2`` (or
  ``$SUBMITIT_PREEMPT_SIGNAL``) at process start. It re-pickles the Hydra task
  and calls ``scontrol requeue`` -- but it never saves a Lightning checkpoint.
* **Lightning** would save a checkpoint and requeue, but its
  ``SignalConnector`` refuses to register on a signal that already has a
  handler, so under submitit it silently does nothing.

Result: every preemption restarts from the last *periodic* ``last.ckpt``.

``PreemptionCheckpoint`` installs its own handler at ``on_fit_start`` that
saves a checkpoint first and then hands the signal to the handler that was
installed before it (submitit's requeue). The saved path defaults to
``<default_root_dir>/checkpoints/last.ckpt``, which is what ``trainHydra.py``
resumes from when ``SLURM_RESTART_COUNT`` is set.

The window available is the partition's preemption ``GraceTime`` (300 s on
``overcap``; ``rl2-lab`` does not preempt), or ``signal_delay_s`` before the
time limit.
"""

from __future__ import annotations

import logging
import os
import signal
from pathlib import Path
from typing import Optional

from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

log = logging.getLogger(__name__)

_SUBMITIT_SIGNAL_ENV = "SUBMITIT_PREEMPT_SIGNAL"


def preempt_signal() -> signal.Signals:
    """The signal submitit listens for: ``$SUBMITIT_PREEMPT_SIGNAL`` or ``USR2``."""
    name = os.environ.get(_SUBMITIT_SIGNAL_ENV, "USR2")
    return signal.Signals[f"SIG{name}"]


class PreemptionCheckpoint(Callback):
    """Save a checkpoint on the Slurm preemption signal, then chain to the
    previously installed handler (submitit's requeue).

    Args:
        ckpt_path: where to save. Defaults to
            ``<trainer.default_root_dir>/checkpoints/last.ckpt``.
        signum: signal to handle. Defaults to :func:`preempt_signal`.
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        signum: Optional[signal.Signals] = None,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.signum = signal.Signals(signum) if signum is not None else preempt_signal()
        self._trainer: Optional[Trainer] = None
        self._previous = None
        self._installed = False
        self._pid: Optional[int] = None

    # -- Callback hooks ------------------------------------------------------

    def on_fit_start(self, trainer: Trainer, pl_module) -> None:
        self._trainer = trainer
        self._pid = os.getpid()
        if self.ckpt_path is None:
            self.ckpt_path = str(
                Path(trainer.default_root_dir) / "checkpoints" / "last.ckpt"
            )
        self._previous = signal.getsignal(self.signum)
        signal.signal(self.signum, self._handle)
        self._installed = True
        if callable(self._previous):
            log.info(
                "PreemptionCheckpoint: on %s save %s, then chain to the existing handler",
                self.signum.name,
                self.ckpt_path,
            )
        else:
            log.warning(
                "PreemptionCheckpoint: on %s save %s; no other handler is installed, "
                "so nothing will requeue this job",
                self.signum.name,
                self.ckpt_path,
            )

    def teardown(self, trainer: Trainer, pl_module, stage: str) -> None:
        if self._installed:
            signal.signal(self.signum, self._previous)
            self._installed = False

    # -- handler -------------------------------------------------------------

    def _handle(self, signum, frame) -> None:
        if os.getpid() != self._pid:
            # Slurm signals every process in the step. Forked DataLoader
            # workers inherit this handler; only the trainer process may act.
            return
        trainer = self._trainer
        assert trainer is not None and self.ckpt_path is not None
        log.warning(
            "Caught %s: saving checkpoint to %s before requeue",
            signal.Signals(signum).name,
            self.ckpt_path,
        )
        try:
            self._claim_last_ckpt(trainer)
            # Write next to the target and rename, so a SIGKILL mid-write
            # (GraceTime expiry) cannot leave a truncated last.ckpt behind.
            tmp = self.ckpt_path + ".preempt"
            # Collective: every rank must get here. Saving from inside the
            # signal handler is the same design as Lightning's own SLURM
            # handler, with the same caveat: a rank interrupted mid-NCCL
            # collective cannot join this one, and the step then hangs until
            # GraceTime's SIGKILL. All shipped launchers run world size 1; a
            # multi-GPU follow-up should flag here and save from
            # on_train_batch_end instead.
            trainer.save_checkpoint(tmp)
            if trainer.is_global_zero:
                os.replace(tmp, self.ckpt_path)
        except Exception:
            # A failed save (quota, NFS) must not cancel the requeue: the job
            # would otherwise end with nothing saved and nothing rescheduled.
            # If only rank 0 fails, the other ranks stay in save_checkpoint's
            # barrier until the requeue tears the step down; acceptable.
            log.exception("Preemption checkpoint failed; requeuing anyway")
        finally:
            if callable(self._previous):
                # submitit's checkpoint_and_try_requeue (it requeues on rank 0
                # only and may raise SystemExit; that propagates, as it would
                # without this callback).
                self._previous(signum, frame)

    def _claim_last_ckpt(self, trainer: Trainer) -> None:
        """Tell every ``ModelCheckpoint(save_last=True)`` writing into the same
        directory that ``last.ckpt`` is its file.

        Otherwise its version counter sees a ``last.ckpt`` it did not write and
        every later periodic save goes to ``last-v1.ckpt``, ``last-v2.ckpt``, ...
        while ``trainHydra`` keeps resuming from the stale ``last.ckpt``. The
        path is part of ``ModelCheckpoint``'s saved state, so this also holds
        in the resumed process. Compared through ``realpath`` because Lightning
        resolves ``dirpath`` that way but not ``default_root_dir``.
        """
        assert self.ckpt_path is not None
        target = Path(self.ckpt_path)
        target_dir = os.path.realpath(target.parent)
        for cb in trainer.checkpoint_callbacks:
            if not (isinstance(cb, ModelCheckpoint) and cb.save_last and cb.dirpath):
                continue
            last_name = cb.CHECKPOINT_NAME_LAST + cb.FILE_EXTENSION
            if os.path.realpath(cb.dirpath) == target_dir and target.name == last_name:
                cb.last_model_path = os.path.join(cb.dirpath, last_name)
