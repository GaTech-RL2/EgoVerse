"""The torch.compile knob: off unless asked, in-place so checkpoints survive."""

import torch.nn as nn
from omegaconf import OmegaConf

from egomimic.pl_utils.pl_model import ModelWrapper


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class _Algo:
    def __init__(self):
        self.nets = nn.ModuleDict({"policy": _Tiny()})

    def compile_targets(self):
        return [self.nets["policy"]]


def _wrapper_with(compile_cfg, monkeypatch):
    algo = _Algo()
    monkeypatch.setattr(ModelWrapper, "_instantiate_model", lambda *a, **k: algo)
    model = {"robomimic_model": {}}
    if compile_cfg is not None:
        model["compile"] = compile_cfg
    ModelWrapper(config_tree=OmegaConf.create({"model": model}), norm_stats_state=None)
    return algo


def _spy_compile(monkeypatch):
    calls = []
    monkeypatch.setattr(nn.Module, "compile", lambda self, **kw: calls.append(kw))
    return calls


def test_compile_is_off_unless_asked(monkeypatch):
    calls = _spy_compile(monkeypatch)
    _wrapper_with(None, monkeypatch)
    _wrapper_with({"enabled": False, "mode": None}, monkeypatch)
    assert calls == []


def test_compile_forwards_mode_and_dynamic(monkeypatch):
    calls = _spy_compile(monkeypatch)
    _wrapper_with(
        {"enabled": True, "mode": "reduce-overhead", "dynamic": True}, monkeypatch
    )
    _wrapper_with({"enabled": True, "mode": None}, monkeypatch)
    assert calls == [{"dynamic": True, "mode": "reduce-overhead"}, {"dynamic": False}]


def test_compile_keeps_state_dict_keys(monkeypatch):
    """nn.Module.compile is in-place; torch.compile(module) would prefix every
    key with `_orig_mod.` and break checkpoints written before the flag."""
    before = list(_Tiny().state_dict())
    algo = _wrapper_with({"enabled": True, "mode": None}, monkeypatch)
    assert list(algo.nets["policy"].state_dict()) == before
