"""The torch.compile knob: off unless asked, in-place so checkpoints survive."""

import torch.nn as nn
from omegaconf import OmegaConf

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.utils.compile_utils import compile_modules


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


def test_compile_keeps_state_dict_keys():
    """nn.Module.compile is in-place; torch.compile(module) would prefix every
    key with `_orig_mod.` and break checkpoints written before the flag."""
    module = _Tiny()
    before = list(module.state_dict())
    compile_modules([("tiny", module)])
    assert list(module.state_dict()) == before


def test_compile_modules_skips_non_modules_and_reports_names():
    """A head without a sub-network hands back None (or something that is not a
    Module); compiling it must be a skip, not a crash."""
    assert compile_modules([("a", _Tiny()), ("b", None), ("c", object())]) == ["a"]


class _SpyAlgo:
    def __init__(self):
        self.calls = []
        self.nets = nn.ModuleDict({"policy": _Tiny()})

    def compile_for_training(self, mode=None, dynamic=False):
        self.calls.append((mode, dynamic))
        return ["policy"]


def _wrapper_with(compile_cfg, monkeypatch):
    algo = _SpyAlgo()
    monkeypatch.setattr(ModelWrapper, "_instantiate_model", lambda *a, **k: algo)
    model = {"robomimic_model": {}}
    if compile_cfg is not None:
        model["compile"] = compile_cfg
    ModelWrapper(config_tree=OmegaConf.create({"model": model}), norm_stats_state=None)
    return algo


def test_compile_is_off_unless_asked(monkeypatch):
    assert _wrapper_with(None, monkeypatch).calls == []
    assert _wrapper_with({"enabled": False, "mode": None}, monkeypatch).calls == []


def test_compile_forwards_mode_and_dynamic(monkeypatch):
    cfg = {"enabled": True, "mode": "reduce-overhead", "dynamic": True}
    assert _wrapper_with(cfg, monkeypatch).calls == [("reduce-overhead", True)]


def test_enable_grad_norm_reaches_the_wrapper():
    """It used to be dropped between the config and ModelWrapper, so a model's
    `enable_grad_norm: false` did nothing."""
    import inspect

    from egomimic import trainHydra

    src = inspect.getsource(trainHydra)
    assert 'enable_grad_norm=cfg.model.get("enable_grad_norm", True)' in src


def test_pi_clips_like_openpi_unless_the_trainer_says_otherwise(compose_resolve):
    from egomimic.trainHydra import _model_trainer_defaults

    cfg = compose_resolve("train_zarr_cartesian", ["model=pi0.5_base"])
    assert cfg.model.gradient_clip_val == 1.0
    assert cfg.model.get("enable_grad_norm", True) is True
    assert _model_trainer_defaults(cfg) == {"gradient_clip_val": 1.0}

    cfg.trainer.gradient_clip_val = 0.5
    assert _model_trainer_defaults(cfg) == {}

    hpt = compose_resolve("train_zarr_cartesian", [])
    assert _model_trainer_defaults(hpt) == {}
