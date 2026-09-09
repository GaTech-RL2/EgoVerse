"""Save-only Slurm signal handling for externally owned requeue workflows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any

from lightning import Callback, LightningModule, Trainer

_SLURM_JOB_ID_RE = re.compile(r"[0-9_]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _exclusive_reservation(path: Path, payload: dict[str, Any]) -> None:
    """Reserve a deterministic publication name with O_EXCL."""

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            descriptor = -1
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def _publish_hardlink_no_overwrite(partial: Path, final: Path) -> None:
    """Atomically publish an existing file without permitting replacement."""

    if not partial.is_file():
        raise FileNotFoundError(f"checkpoint partial is unavailable: {partial}")
    os.link(partial, final)
    _fsync_directory(final.parent)


def _atomic_json_no_overwrite(path: Path, payload: dict[str, Any]) -> None:
    """Publish JSON atomically while refusing to replace an existing proof."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite signal checkpoint proof: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("x") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link is an atomic create-if-absent operation. Unlike replace(),
        # it cannot silently overwrite a proof from another process.
        _publish_hardlink_no_overwrite(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SaveOnlySignalCheckpoint(Callback):
    """Save a full-state checkpoint on SIGUSR2 without invoking requeue.

    The signal handler only increments a Python-side generation counter. The
    request is serviced at a safe Lightning hook. The external runner delivers
    SIGUSR2 to every active Slurm job step, so a rank with no local request does
    no collective work. Once locally requested, every rank calls
    ``Trainer.save_checkpoint`` as required by distributed checkpoint
    strategies; only global rank zero publishes the immutable checkpoint and
    runtime proof.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_signal: signal.Signals = signal.SIGUSR2,
    ) -> None:
        super().__init__()
        checkpoint_dir = Path(checkpoint_dir).expanduser()
        if not checkpoint_dir.is_absolute():
            raise ValueError("signal checkpoint directory must be absolute")
        if save_signal != signal.SIGUSR2:
            raise ValueError("external save-only checkpoint signal must be SIGUSR2")
        self.checkpoint_dir = checkpoint_dir.resolve()
        self.save_signal = save_signal
        self._request_generation = 0
        self._handled_generation = 0
        self._save_index = 0
        self._installed_handler: Any = None
        self._previous_handler: Any = None

    def _request_checkpoint(self, _signum: int, _frame: FrameType | None) -> None:
        self._request_generation += 1

    def setup(
        self,
        _trainer: Trainer,
        _pl_module: LightningModule,
        stage: str,
    ) -> None:
        if stage != "fit" or self._installed_handler is not None:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "save-only signal handler must be installed on main thread"
            )
        previous = signal.getsignal(self.save_signal)
        if previous != signal.SIG_DFL:
            raise RuntimeError(
                f"refusing to replace existing {self.save_signal.name} handler: "
                f"{previous!r}"
            )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._previous_handler = previous
        self._installed_handler = self._request_checkpoint
        signal.signal(self.save_signal, self._installed_handler)

    def teardown(
        self,
        _trainer: Trainer,
        _pl_module: LightningModule,
        stage: str,
    ) -> None:
        if stage != "fit" or self._installed_handler is None:
            return
        if signal.getsignal(self.save_signal) is self._installed_handler:
            signal.signal(self.save_signal, self._previous_handler)
        self._installed_handler = None
        self._previous_handler = None

    def _checkpoint_path(self, trainer: Trainer) -> tuple[Path, str, int]:
        job_id = os.environ.get("SLURM_JOB_ID", "")
        if _SLURM_JOB_ID_RE.fullmatch(job_id) is None:
            raise RuntimeError(f"invalid or missing SLURM_JOB_ID: {job_id!r}")
        restart_text = os.environ.get("SLURM_RESTART_COUNT", "0")
        try:
            restart_count = int(restart_text)
        except ValueError as exc:
            raise RuntimeError(
                f"SLURM_RESTART_COUNT must be an integer: {restart_text!r}"
            ) from exc
        if restart_count < 0:
            raise RuntimeError("SLURM_RESTART_COUNT must be nonnegative")
        path = self.checkpoint_dir / (
            f"signal-job={job_id}-restart={restart_count:03d}-"
            f"request={self._save_index:03d}-epoch={int(trainer.current_epoch)}-"
            f"step={int(trainer.global_step)}.ckpt"
        )
        return path, job_id, restart_count

    def _service(self, trainer: Trainer, hook: str) -> None:
        observed_generation = self._request_generation
        local_requested = observed_generation > self._handled_generation
        if not local_requested:
            return
        # Advance only to the generation observed before checkpoint work. A
        # second signal arriving during the save remains pending for the next
        # safe hook instead of being lost.
        self._handled_generation = observed_generation
        checkpoint_path, job_id, restart_count = self._checkpoint_path(trainer)
        partial_path = checkpoint_path.with_name(f".{checkpoint_path.name}.partial")
        reservation_path = checkpoint_path.with_name(f".{checkpoint_path.name}.reserve")
        proof_path = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".save-only.json"
        )

        reservation_error = ""
        if trainer.is_global_zero:
            try:
                collisions = [
                    path
                    for path in (
                        checkpoint_path,
                        partial_path,
                        reservation_path,
                        proof_path,
                    )
                    if path.exists()
                ]
                if collisions:
                    raise FileExistsError(
                        "refusing to overwrite signal checkpoint artifacts: "
                        + ", ".join(map(str, collisions))
                    )
                _exclusive_reservation(
                    reservation_path,
                    {
                        "schema_version": 1,
                        "checkpoint_path": str(checkpoint_path),
                        "job_id": job_id,
                        "restart_count": restart_count,
                        "request_index": self._save_index,
                    },
                )
            except Exception as exc:
                reservation_error = f"{type(exc).__name__}: {exc}"
        reservation_error = trainer.strategy.broadcast(reservation_error, src=0)
        if reservation_error:
            raise RuntimeError(
                f"failed to reserve signal checkpoint publication: {reservation_error}"
            )

        # Lightning requires every process to call this method. weights_only=False
        # preserves optimizer, scheduler, EMA, loop/data state, and global step.
        # The framework writes only to a hidden partial. Rank zero publishes the
        # final name later through a create-if-absent hard link.
        trainer.save_checkpoint(str(partial_path), weights_only=False)

        publication_error = ""
        if trainer.is_global_zero:
            try:
                partial_stat = partial_path.stat()
                checkpoint_sha256 = _sha256(partial_path)
                _publish_hardlink_no_overwrite(partial_path, checkpoint_path)
                partial_path.unlink()
                reservation_path.unlink()
                _fsync_directory(checkpoint_path.parent)
                if partial_path.exists() or reservation_path.exists():
                    raise RuntimeError(
                        "partial or reservation remains after checkpoint publication"
                    )
                _atomic_json_no_overwrite(
                    proof_path,
                    {
                        "schema_version": 1,
                        "status": "FULL_STATE_CHECKPOINT_PUBLISHED_UNVALIDATED",
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_sha256,
                        "checkpoint_size_bytes": partial_stat.st_size,
                        "epoch": int(trainer.current_epoch),
                        "global_step": int(trainer.global_step),
                        "hook": hook,
                        "job_id": job_id,
                        "restart_count": restart_count,
                        "save_signal": self.save_signal.name,
                        "world_size": int(trainer.strategy.world_size),
                        "all_ranks_save_checkpoint_returned": True,
                        "weights_only": False,
                        "publish_atomic_no_overwrite": True,
                        "partial_path": str(partial_path),
                        "partial_path_absent": True,
                        "reservation_path": str(reservation_path),
                        "reservation_path_absent": True,
                        "application_validation_status": "NOT_RUN_BY_CALLBACK",
                        "scontrol_invoked": False,
                        "wandb_finalize_invoked": False,
                        "saved_at_unix_ns": time.time_ns(),
                    },
                )
            except Exception as exc:  # propagated to every rank below
                errors = [f"{type(exc).__name__}: {exc}"]
                for stale_path in (partial_path, reservation_path):
                    try:
                        stale_path.unlink(missing_ok=True)
                    except Exception as cleanup_exc:
                        errors.append(
                            f"cleanup {stale_path}: "
                            f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                        )
                try:
                    _fsync_directory(checkpoint_path.parent)
                except Exception as cleanup_exc:
                    errors.append(
                        "cleanup directory fsync: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
                publication_error = "; ".join(errors)
        publication_error = trainer.strategy.broadcast(publication_error, src=0)
        if publication_error:
            raise RuntimeError(
                f"failed to publish signal checkpoint atomically: {publication_error}"
            )

        self._save_index += 1

    def on_train_batch_end(
        self,
        trainer: Trainer,
        _pl_module: LightningModule,
        _outputs: Any,
        _batch: Any,
        _batch_idx: int,
    ) -> None:
        self._service(trainer, "on_train_batch_end")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        _pl_module: LightningModule,
        _outputs: Any,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> None:
        self._service(trainer, "on_validation_batch_end")

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        _pl_module: LightningModule,
    ) -> None:
        self._service(trainer, "on_train_epoch_end")

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        _pl_module: LightningModule,
    ) -> None:
        self._service(trainer, "on_validation_epoch_end")

    def on_fit_end(
        self,
        trainer: Trainer,
        _pl_module: LightningModule,
    ) -> None:
        self._service(trainer, "on_fit_end")
