"""Lightning callback for the non-production ICE requeue integration proof.

The callback requests the Slurm batch boundary after exactly five optimizer
steps, waits until every training rank has observed the runner-forwarded
SIGUSR2 request, and prevents another optimizer step from starting while the
generic runner publishes and authenticates the full-state checkpoint.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from lightning import Callback, LightningModule, Trainer

from egomimic.utils.slurm_requeue import SaveOnlySignalCheckpoint


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class FixedStepRequeueBoundary(Callback):
    """Request runner-owned requeue after exactly five optimizer steps."""

    def __init__(
        self,
        *,
        initial_global_step: int,
        trigger_global_step: int,
        slurm_job_id: str,
        scancel: str,
        record_path: str,
        scancel_timeout_seconds: float = 10.0,
        signal_ack_timeout_seconds: float = 30.0,
        requeue_wait_timeout_seconds: float = 300.0,
    ) -> None:
        super().__init__()
        if isinstance(initial_global_step, bool) or not isinstance(
            initial_global_step, int
        ):
            raise TypeError("initial_global_step must be an integer")
        if isinstance(trigger_global_step, bool) or not isinstance(
            trigger_global_step, int
        ):
            raise TypeError("trigger_global_step must be an integer")
        if trigger_global_step - initial_global_step != 5:
            raise ValueError("integration boundary must be exactly five optimizer steps")
        if re.fullmatch(r"[1-9][0-9_]*", slurm_job_id) is None:
            raise ValueError("slurm_job_id is invalid")
        scancel_path = Path(scancel)
        if (
            not scancel_path.is_absolute()
            or not scancel_path.is_file()
            or not os.access(scancel_path, os.X_OK)
        ):
            raise ValueError("scancel must name an executable absolute file")
        record = Path(record_path)
        if not record.is_absolute():
            raise ValueError("record_path must be absolute")
        if min(
            scancel_timeout_seconds,
            signal_ack_timeout_seconds,
            requeue_wait_timeout_seconds,
        ) <= 0:
            raise ValueError("fixed-step boundary timeouts must be positive")

        self.initial_global_step = initial_global_step
        self.trigger_global_step = trigger_global_step
        self.slurm_job_id = slurm_job_id
        self.scancel = str(scancel_path.resolve())
        self.record_path = record.resolve()
        self.scancel_timeout_seconds = float(scancel_timeout_seconds)
        self.signal_ack_timeout_seconds = float(signal_ack_timeout_seconds)
        self.requeue_wait_timeout_seconds = float(requeue_wait_timeout_seconds)
        self._sent = False
        self._save_callback: SaveOnlySignalCheckpoint | None = None

    def setup(
        self, trainer: Trainer, _pl_module: LightningModule, stage: str
    ) -> None:
        if stage != "fit":
            return
        matches = [
            callback
            for callback in trainer.callbacks
            if isinstance(callback, SaveOnlySignalCheckpoint)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "fixed-step integration requires exactly one save-only callback"
            )
        save_callback = matches[0]
        if trainer.callbacks.index(self) >= trainer.callbacks.index(save_callback):
            raise RuntimeError(
                "fixed-step boundary must precede the save-only callback"
            )
        self._save_callback = save_callback

    def on_train_batch_end(
        self,
        trainer: Trainer,
        _pl_module: LightningModule,
        _outputs: Any,
        _batch: Any,
        _batch_idx: int,
    ) -> None:
        if self._sent:
            return
        observed_step = int(trainer.global_step)
        if observed_step < self.trigger_global_step:
            return
        if observed_step != self.trigger_global_step:
            raise RuntimeError(
                "fixed-step integration boundary was missed: "
                f"expected {self.trigger_global_step}, observed {observed_step}"
            )
        if os.environ.get("ICE_REQUEUE_OWNER") != "runner":
            raise RuntimeError("the generic runner must be the sole requeue owner")
        if os.environ.get("ICE_CHILD_REQUEUE_DISABLED") != "1":
            raise RuntimeError("framework requeue must be disabled")
        if os.environ.get("SLURM_JOB_ID") != self.slurm_job_id:
            raise RuntimeError("live Slurm job differs from the boundary contract")
        if os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
            raise RuntimeError("fixed-step boundary may run only on restart zero")
        if self._save_callback is None:
            raise RuntimeError("fixed-step boundary callback was not set up")

        world_size = int(trainer.strategy.world_size)
        if world_size < 1:
            raise RuntimeError("trainer world size must be positive")
        if trainer.is_global_zero and self.record_path.exists():
            raise RuntimeError("fixed-step boundary proof already exists")

        self._sent = True
        generation_before = self._save_callback._request_generation
        trainer.strategy.barrier("unite-fixed-step-boundary-ready")
        argv = [self.scancel, "--batch", "--signal=USR1", self.slurm_job_id]
        signal_environment = os.environ.copy()
        for name in tuple(signal_environment):
            if name.startswith("SCANCEL_") or name == "SLURM_CLUSTERS":
                signal_environment.pop(name)

        completed: subprocess.CompletedProcess[str] | None = None
        command_error = ""
        if trainer.is_global_zero:
            try:
                completed = subprocess.run(
                    argv,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=signal_environment,
                    check=False,
                    timeout=self.scancel_timeout_seconds,
                )
                if completed.returncode != 0:
                    command_error = (
                        "fixed-step Slurm boundary request failed: "
                        f"returncode={completed.returncode}, "
                        f"stderr={completed.stderr!r}"
                    )
            except subprocess.TimeoutExpired:
                command_error = "fixed-step Slurm boundary request timed out"
        command_error = trainer.strategy.broadcast(command_error, src=0)
        if command_error:
            raise RuntimeError(command_error)

        wait_started = time.monotonic()
        deadline = wait_started + self.signal_ack_timeout_seconds
        while self._save_callback._request_generation <= generation_before:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "runner did not forward SIGUSR2 to every rank before timeout"
                )
            time.sleep(0.05)
        generation_after = self._save_callback._request_generation
        trainer.strategy.barrier("unite-fixed-step-boundary-signal-acknowledged")

        if trainer.is_global_zero:
            if completed is None:
                raise AssertionError("global zero has no boundary command result")
            _atomic_json(
                self.record_path,
                {
                    "schema_version": 1,
                    "status": "ALL_RANKS_SAVE_SIGNAL_ACKNOWLEDGED",
                    "job_id": self.slurm_job_id,
                    "restart_count": 0,
                    "initial_global_step": self.initial_global_step,
                    "trigger_global_step": self.trigger_global_step,
                    "observed_global_step": observed_step,
                    "optimizer_steps_advanced": 5,
                    "step_after_signal_executed": False,
                    "signal": "SIGUSR1",
                    "signal_target": "slurm_batch_step",
                    "save_signal": "SIGUSR2",
                    "save_request_generation_before": generation_before,
                    "save_request_generation_after": generation_after,
                    "save_acknowledged_ranks": list(range(world_size)),
                    "world_size": world_size,
                    "save_signal_ack_wait_seconds": time.monotonic() - wait_started,
                    "argv": argv,
                    "scancel_stdout": completed.stdout,
                    "scancel_stderr": completed.stderr,
                    "requested_at_unix": time.time(),
                },
            )

    def on_train_batch_start(
        self,
        _trainer: Trainer,
        _pl_module: LightningModule,
        _batch: Any,
        _batch_idx: int,
    ) -> None:
        if not self._sent:
            return
        deadline = time.monotonic() + self.requeue_wait_timeout_seconds
        while time.monotonic() < deadline:
            time.sleep(0.25)
        raise RuntimeError(
            "Slurm did not terminate the fixed-step attempt after checkpoint save"
        )
