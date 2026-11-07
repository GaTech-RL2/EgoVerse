"""Explicit checkpoint clocks and full-state publication for separate opscale roles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checkpoint_metadata(path):
    """Inspect trusted local full-state checkpoints without materializing tensors."""
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    required = ("state_dict", "optimizer_states", "lr_schedulers", "loops")
    if any(not value.get(key) for key in required):
        raise ValueError(f"Checkpoint lacks full training state: {path}")
    epoch, step = int(value["epoch"]), int(value["global_step"])
    if epoch < 0 or step <= 0 or step != (epoch + 1) * 100:
        raise ValueError("Checkpoint is not an exact 100-update epoch boundary")
    if value.get("ema") is not None:
        raise ValueError("This recipe requires raw weights and EMA disabled")
    return {
        "epoch_zero_based": epoch,
        "completed_epochs": epoch + 1,
        "global_step": step,
        "ema_present": False,
        **{f"{key}_present": True for key in required},
    }


class CheckpointEvaluationTrainer(Trainer):
    """Evaluation-only Trainer whose public clock names the loaded checkpoint.

    Lightning's validate() deliberately leaves fit-loop counters at zero. The
    existing evaluators, video schedule, captions and W&B all read these public
    properties. Override them explicitly, rather than editing private fit-loop
    state or pretending that validate() restores an optimizer.
    """

    def __init__(self, *, checkpoint_epoch: int, checkpoint_step: int, **kwargs):
        self._checkpoint_epoch = int(checkpoint_epoch)
        self._checkpoint_step = int(checkpoint_step)
        if self._checkpoint_step != (self._checkpoint_epoch + 1) * 100:
            raise ValueError("Checkpoint clock is inconsistent")
        super().__init__(**kwargs)

    @property
    def current_epoch(self):
        return self._checkpoint_epoch

    @property
    def global_step(self):
        return self._checkpoint_step

    def fit(self, *args, **kwargs):
        raise RuntimeError("CheckpointEvaluationTrainer cannot train")


class CheckpointIdentity(Callback):
    """Log the SHA alongside the checkpoint clock, including same-step aliases."""

    def __init__(self, *, checkpoint_sha256, split_sha256, source_wandb_id):
        self.values = {
            "checkpoint/sha256": checkpoint_sha256,
            "checkpoint/split_sha256": split_sha256,
            "checkpoint/source_wandb_id": source_wandb_id,
        }

    def on_validation_start(self, trainer, pl_module):
        values = {
            **self.values,
            "checkpoint/global_step": trainer.global_step,
            "checkpoint/completed_epochs": trainer.current_epoch + 1,
        }
        for logger in trainer.loggers:
            logger.log_metrics(values, step=trainer.global_step)


class PublishedCheckpoint(ModelCheckpoint):
    """Publish READY only after the full atomic Lightning save and SHA finish.

    All epoch checkpoints are retained. A last.ckpt symlink is only a resume
    convenience and is never queued as a second checkpoint.
    """

    def __init__(
        self,
        *,
        cell,
        source_wandb_id,
        training_wandb_id,
        split_sha256,
        queue_dir,
        **kwargs,
    ):
        expected = {
            "save_top_k": -1,
            "save_last": "link",
            "every_n_epochs": 50,
            "save_on_train_epoch_end": True,
            "save_weights_only": False,
            "monitor": None,
        }
        for key, value in expected.items():
            if kwargs.get(key, value) != value:
                raise ValueError(f"{key} must equal {value!r}")
            kwargs[key] = value
        super().__init__(**kwargs)
        self.cell = cell
        self.source_wandb_id = source_wandb_id
        self.training_wandb_id = training_wandb_id
        self.split_sha256 = split_sha256
        self.queue_dir = Path(queue_dir)

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if not trainer.is_global_zero:
            return
        path = Path(filepath).resolve()
        metadata = checkpoint_metadata(path)
        if metadata["global_step"] != trainer.global_step:
            raise RuntimeError("Saved checkpoint does not match training clock")
        if metadata["completed_epochs"] % 50:
            raise RuntimeError("Unexpected checkpoint cadence")
        digest = sha256(path)
        record = {
            "schema_version": 1,
            "cell": self.cell,
            "source_wandb_id": self.source_wandb_id,
            "training_wandb_id": self.training_wandb_id,
            "split_sha256": self.split_sha256,
            "sha256": digest,
            "size": path.stat().st_size,
            "local_path": str(path),
            "checkpoint": metadata,
            "ready_after_verification": True,
        }
        target = self.queue_dir / "ready" / f"{trainer.global_step:09d}-{digest}.json"
        if target.exists() and json.loads(target.read_text()) != record:
            raise RuntimeError("Conflicting READY checkpoint identity")
        if not target.exists():
            atomic_json(target, record)

    def on_fit_end(self, trainer, pl_module):
        super().on_fit_end(trainer, pl_module)
        if not trainer.is_global_zero:
            return
        if trainer.global_step != 400000:
            raise RuntimeError("Training returned before the prescribed 400000 updates")
        ready = [
            json.loads(p.read_text()) for p in (self.queue_dir / "ready").glob("*.json")
        ]
        matching = [
            x for x in ready if x["checkpoint"]["global_step"] == trainer.global_step
        ]
        if len(matching) != 1:
            raise RuntimeError("Final full-state checkpoint has no unique READY record")
        atomic_json(
            self.queue_dir / "training_complete.json",
            {
                "schema_version": 1,
                "cell": self.cell,
                "completed": True,
                "global_step": trainer.global_step,
                "checkpoint_sha256": matching[0]["sha256"],
                "training_wandb_id": self.training_wandb_id,
            },
        )
