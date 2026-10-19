"""Validation videos written by ``EvalVideo`` are also logged to wandb."""

from __future__ import annotations

import types
from pathlib import Path

import torch
from lightning.pytorch.loggers import WandbLogger

from egomimic.eval.eval_video import EvalVideo

HUMAN_BIMANUAL = 3


class _Eval(EvalVideo):
    def __init__(self, root):
        super().__init__()
        self._root = root

    def root_dir(self):
        return self._root

    def compute_metrics_and_viz(self, batch, do_viz=True):
        return {}, {}


def _trainer(loggers, epoch=99, step=1234):
    return types.SimpleNamespace(
        is_global_zero=True, current_epoch=epoch, global_step=step, loggers=loggers
    )


def _frames():
    return torch.randint(0, 255, (8, 32, 32, 3), dtype=torch.uint8)


def test_video_logged_to_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    logger = WandbLogger(save_dir=str(tmp_path), offline=True, project="t")
    ev = _Eval(str(tmp_path / "run"))
    ev.trainer = _trainer([logger])
    ev.val_counter[HUMAN_BIMANUAL] = 0

    ev._write_video(HUMAN_BIMANUAL, _frames())
    run_dir = Path(logger.experiment.dir)
    logger.experiment.finish()

    local = tmp_path / "run/videos/epoch_99/HUMAN_BIMANUAL/validation_video_0.mp4"
    assert local.exists()
    media = list((run_dir / "media/videos/videos").glob("HUMAN_BIMANUAL_*.mp4"))
    assert len(media) == 1


def test_no_wandb_logger_is_a_noop(tmp_path):
    ev = _Eval(str(tmp_path))
    ev.trainer = _trainer([])
    ev.val_counter[HUMAN_BIMANUAL] = 0
    ev._write_video(HUMAN_BIMANUAL, _frames())
    assert (tmp_path / "videos/epoch_99/HUMAN_BIMANUAL/validation_video_0.mp4").exists()


def test_prefixed_head_logs_and_writes_under_its_prefix(tmp_path):
    """The train_viz / opsplit heads: ``<prefix>/...`` keys, ``videos_<prefix>/``."""
    logged = []
    for prefix, key, videos in (
        (None, "Valid/x", "videos"),
        ("train_viz", "train_viz/Valid/x", "videos_train_viz"),
    ):
        ev = _Eval(str(tmp_path))
        ev.prefix = prefix
        ev.compute_metrics_and_viz = lambda batch, do_viz=True: ({"Valid/x": 1.0}, {})
        ev.trainer = _trainer([])
        ev.trainer.lightning_module = types.SimpleNamespace(
            device="cpu", log_dict=lambda metrics, **kw: logged.append(set(metrics))
        )
        ev.on_validation_step({}, 0, mode="metrics")
        assert logged[-1] == {key}
        ev.val_counter[HUMAN_BIMANUAL] = 0
        ev._write_video(HUMAN_BIMANUAL, _frames())
        assert (tmp_path / videos / "epoch_99/HUMAN_BIMANUAL").is_dir()
