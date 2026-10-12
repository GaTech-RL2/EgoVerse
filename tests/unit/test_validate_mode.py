"""mode=validate: score one checkpoint through the training val heads on one
device, with the EMA weights whenever the checkpoint has them."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from fixtures.recipes import (
    RECIPES,
    common_overrides,
    cpu_trainer_overrides,
    hpt_small_overrides,
)
from fixtures.train_harness import compose_recipe, hermetic_env, write_fixtures
from omegaconf import OmegaConf

import egomimic.trainHydra as th
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.utils.checkpoint_utils import load_checkpoint_weights


class _Algo:
    def __init__(self):
        self.nets = nn.ModuleDict({"policy": nn.Linear(2, 2)})


def _wrapper(monkeypatch, model_cfg):
    monkeypatch.setattr(ModelWrapper, "_instantiate_model", lambda *a, **k: _Algo())
    return ModelWrapper(
        config_tree=OmegaConf.create({"model": model_cfg}), norm_stats_state=None
    )


def _save_ckpt(monkeypatch, path, raw=1.0, ema=None):
    src = _wrapper(monkeypatch, {"robomimic_model": {}, "ema": {"enabled": True}})
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(raw)
    checkpoint = {"state_dict": src.state_dict()}
    if ema is not None:
        src._ensure_ema()
        for shadow in src.ema.shadow:
            shadow.fill_(ema)
        src.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, path)
    return path


def _validate_cfg(ckpt, model_cfg):
    return OmegaConf.create(
        {
            "mode": "validate",
            "ckpt_path": str(ckpt),
            "model": model_cfg,
            "trainer": {"devices": 4, "num_nodes": 2, "limit_val_batches": 5},
            "logger": {"wandb": {}},
        }
    )


def _weights(wrapper):
    return {float(v) for p in wrapper.parameters() for v in p.detach().flatten()}


@pytest.mark.parametrize("ema_cfg", [None, {"enabled": False}, {"enabled": True}])
def test_ema_checkpoint_is_scored_with_ema_weights(monkeypatch, tmp_path, ema_cfg):
    """The #710 port loaded the raw state_dict only: a config without EMA
    enabled scored the raw weights of a run that validated its EMA."""
    ckpt = _save_ckpt(monkeypatch, tmp_path / "a.ckpt", raw=1.0, ema=3.0)
    model_cfg = {"robomimic_model": {}}
    if ema_cfg is not None:
        model_cfg["ema"] = ema_cfg
    cfg = _validate_cfg(ckpt, model_cfg)

    assert th._enable_checkpoint_ema(cfg) is True
    wrapper = _wrapper(monkeypatch, OmegaConf.to_container(cfg.model))
    load_checkpoint_weights(wrapper, ckpt)
    assert _weights(wrapper) == {1.0}
    wrapper.on_validation_start()
    assert _weights(wrapper) == {3.0}


def test_raw_checkpoint_is_scored_raw(monkeypatch, tmp_path):
    ckpt = _save_ckpt(monkeypatch, tmp_path / "a.ckpt", raw=2.0)
    cfg = _validate_cfg(ckpt, {"robomimic_model": {}})

    assert th._enable_checkpoint_ema(cfg) is False
    assert cfg.model.get("ema") is None
    wrapper = _wrapper(monkeypatch, OmegaConf.to_container(cfg.model))
    load_checkpoint_weights(wrapper, ckpt)
    wrapper.on_validation_start()
    assert wrapper.ema is None
    assert _weights(wrapper) == {2.0}


@pytest.mark.parametrize("ckpt_path", [None, "last", "/nonexistent/x.ckpt"])
def test_validate_needs_a_checkpoint_file(ckpt_path):
    cfg = _validate_cfg("x", {"robomimic_model": {}})
    cfg.ckpt_path = ckpt_path
    with pytest.raises(ValueError, match="ckpt_path"):
        th._enable_checkpoint_ema(cfg)


def test_one_device_reads_every_frame_the_training_ranks_scored():
    cfg = _validate_cfg("x", {})
    th._single_device_validate_overrides(cfg)
    assert cfg.trainer.limit_val_batches == 5 * 4 * 2
    assert (cfg.trainer.devices, cfg.trainer.num_nodes) == (1, 1)
    assert cfg.trainer.strategy == "auto"
    assert cfg.trainer.num_sanity_val_steps == 0
    assert cfg.logger is None


def test_validate_keeps_the_training_val_heads():
    cfg = OmegaConf.create(
        {
            "mode": "validate",
            "evaluator": {"_target_": "x"},
            "data": {"unseen_op_valid_datasets": {"human_bimanual": None}},
        }
    )
    assert th._train_viz_enabled(cfg)
    assert th._config_val_datasets(cfg, "unseen_op_valid", instantiate=None) == {
        "human_bimanual": None
    }
    cfg.mode = "eval"
    assert not th._train_viz_enabled(cfg)
    assert th._config_val_datasets(cfg, "unseen_op_valid", instantiate=None) == {}


def test_validate_checkpoint_writes_metrics(monkeypatch, tmp_path):
    ckpt = _save_ckpt(monkeypatch, tmp_path / "a.ckpt", raw=1.0, ema=3.0)
    cfg = _validate_cfg(ckpt, {"robomimic_model": {}})
    th._enable_checkpoint_ema(cfg)
    wrapper = _wrapper(monkeypatch, OmegaConf.to_container(cfg.model))
    seen = {}

    def validate(model, datamodule):
        model.on_validation_start()
        seen["weights"] = _weights(model)
        return [{"Valid/mse": torch.tensor(0.25)}, {"train_viz/mse": 0.5}]

    trainer = SimpleNamespace(validate=validate, default_root_dir=str(tmp_path))
    path = th._validate_checkpoint(cfg, trainer, wrapper, datamodule=None)

    assert seen["weights"] == {3.0}
    out = json.loads(open(path).read())
    assert out == {
        "ckpt_path": str(ckpt),
        "ema": True,
        "metrics": [{"Valid/mse": 0.25}, {"train_viz/mse": 0.5}],
    }


def test_validate_mode_end_to_end(tmp_path, monkeypatch):
    """train() on synthetic episodes: save an EMA checkpoint, then validate it
    under a config without EMA; the val pass sees the EMA weights, no step runs."""
    hermetic_env(monkeypatch)
    recipe = RECIPES[("aria", "hpt")]
    data, out, hashes = write_fixtures(tmp_path, "aria")

    def cfg_for(run_dir, *extra):
        run_dir.mkdir(exist_ok=True)
        return compose_recipe(
            recipe,
            common_overrides(
                recipe.embodiment,
                data,
                run_dir,
                batch_size=2,
                num_workers=0,
                episode_hashes=hashes,
            )
            + cpu_trainer_overrides(2)
            + hpt_small_overrides(recipe.embodiment)
            + list(extra),
            run_dir,
        )

    _, objects = th.train(cfg_for(out, "+model.ema={enabled: true}"))
    ckpt = tmp_path / "trained.ckpt"
    objects["trainer"].save_checkpoint(str(ckpt))
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    for shadow in checkpoint["ema"]["shadow"].values():
        shadow.fill_(0.5)
    torch.save(checkpoint, ckpt)

    scored = []
    monkeypatch.setattr(
        ModelWrapper,
        "validation_step",
        lambda self, *a, **k: scored.append(
            all(bool((p == 0.5).all()) for p in self.ema.params)
        ),
    )
    _, objects = th.train(
        cfg_for(
            tmp_path / "validate",
            "mode=validate",
            f"+ckpt_path={ckpt}",
            "trainer.limit_val_batches=1",
        )
    )

    assert scored == [True]
    assert objects["trainer"].global_step == 0
    result = json.loads((tmp_path / "validate" / "validation_metrics.json").read_text())
    assert result["ema"] is True and result["ckpt_path"] == str(ckpt)
