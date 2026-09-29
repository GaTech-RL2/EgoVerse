"""Weights-only fine-tune init: matched tensors load, a new head stays fresh,
an unrelated checkpoint fails loudly, and a resume ignores it."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import egomimic.trainHydra as train_hydra
from egomimic.utils.checkpoint_utils import init_weights_from_checkpoint


class _Net(nn.Module):
    def __init__(self, head_out: int):
        super().__init__()
        self.trunk = nn.Linear(64, 64)
        self.head = nn.Linear(64, head_out)


def _save(model: nn.Module, path) -> str:
    torch.save({"state_dict": model.state_dict(), "optimizer_states": [1]}, path)
    return str(path)


def test_matched_tensors_load_and_the_new_head_stays_fresh(tmp_path):
    base = _Net(head_out=18)
    ckpt = _save(base, tmp_path / "base.ckpt")
    target = _Net(head_out=14)
    fresh_head = target.head.weight.detach().clone()
    init_weights_from_checkpoint(target, ckpt)
    assert torch.equal(target.trunk.weight, base.trunk.weight)
    assert torch.equal(target.head.weight, fresh_head)


def test_an_unrelated_checkpoint_raises(tmp_path):
    other = nn.Sequential(nn.Linear(3, 3))
    ckpt = _save(other, tmp_path / "other.ckpt")
    with pytest.raises(RuntimeError, match="matched"):
        init_weights_from_checkpoint(_Net(14), ckpt)


def test_resume_ignores_init_weights(tmp_path):
    ckpt = _save(_Net(18), tmp_path / "base.ckpt")
    target = _Net(14)
    before = target.trunk.weight.detach().clone()
    cfg = OmegaConf.create(
        {"init_weights_ckpt": ckpt, "ckpt_path": str(tmp_path / "last.ckpt")}
    )
    assert train_hydra._apply_init_weights(cfg, target) is None
    assert torch.equal(target.trunk.weight, before)


def test_missing_init_checkpoint_raises(tmp_path):
    cfg = OmegaConf.create(
        {"init_weights_ckpt": str(tmp_path / "nope.ckpt"), "ckpt_path": None}
    )
    with pytest.raises(FileNotFoundError):
        train_hydra._apply_init_weights(cfg, _Net(14))


def test_train_config_declares_the_knob(compose_resolve):
    cfg = compose_resolve("train_zarr_cartesian", [])
    assert cfg.init_weights_ckpt is None
