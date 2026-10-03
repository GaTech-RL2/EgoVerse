"""Per-embodiment widths are declared once under ``robomimic_model.dims`` and every
stem/head width in an HPT model config is an interpolation of them (Phase 3).

Checked on every ``model/*.yaml`` whose ``robomimic_model`` is HPT: overriding
each ``dims`` value with a distinct number moves every width bound to it (a
leftover literal or a proprio/action mix-up fails), and every value reaches at
least one width (none is declared but unread).
"""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

import egomimic

# Needed at import time for parametrize (see test_hydra_configs.py).
MODEL_DIR = Path(egomimic.__file__).parent / "hydra_configs" / "model"


def _target(name):
    """``robomimic_model._target_`` of a model config, following its defaults."""
    cfg = yaml.safe_load((MODEL_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    target = cfg.get("robomimic_model", {}).get("_target_")
    for parent in cfg.get("defaults", []):
        if target is None and parent != "_self_":
            target = _target(parent)
    return target


HPT_CONFIGS = sorted(
    p.stem
    for p in MODEL_DIR.glob("*.yaml")
    if _target(p.stem) == "egomimic.algo.hpt.HPT" and p.stem != "egobridge"
)


def _model(compose_resolve, name, extra=()):
    cfg = compose_resolve("train_zarr_cartesian", [f"model={name}", *extra])
    return cfg.model.robomimic_model


def _width_leaves(rm):
    """Yield (path, value, dims_key) for every width leaf of a resolved
    robomimic_model container; dims_key is the ``dims`` entry the leaf
    should read: ``<emb>.proprio``, ``<emb>.action`` or ``action_width``."""
    for emb, stems in (rm.get("stem_specs") or {}).items():
        for key, stem in (stems or {}).items():
            if stem is None:
                continue  # a child config dropping a base stem
            if key.startswith("state_"):
                path = f"stem_specs.{emb}.{key}.input_dim"
                yield path, stem["input_dim"], f"{emb}.proprio"
    for head_name, head in (rm.get("head_specs") or {}).items():
        if head is None:
            continue
        for emb, width in (head.get("infer_ac_dims") or {}).items():
            yield f"head_specs.{head_name}.infer_ac_dims.{emb}", width, f"{emb}.action"
        if "model" in head and "act_dim" in head["model"]:
            shared = head_name == "shared"
            key = "action_width" if shared else f"{head_name}.action"
            yield f"head_specs.{head_name}.model.act_dim", head["model"]["act_dim"], key
        if "output_dim" in head:
            path = f"head_specs.{head_name}.output_dim"
            yield path, head["output_dim"], f"{head_name}.action"


@pytest.mark.parametrize("name", HPT_CONFIGS)
def test_every_dims_value_moves_its_widths(name, compose_resolve):
    dims = OmegaConf.to_container(_model(compose_resolve, name).dims)
    keys = [k for k, v in dims.items() if not isinstance(v, dict)] + [
        f"{emb}.{kind}" for emb, v in dims.items() if isinstance(v, dict) for kind in v
    ]
    distinct = {key: 101 + i for i, key in enumerate(keys)}
    overrides = [f"model.robomimic_model.dims.{k}={v}" for k, v in distinct.items()]
    rm = _model(compose_resolve, name, overrides)
    leaves = list(_width_leaves(OmegaConf.to_container(rm)))
    for path, value, key in leaves:
        assert value == distinct[key], path
    seen = {value for _path, value, _key in leaves}
    dead = [key for key, value in distinct.items() if value not in seen]
    assert not dead, f"dims values no width reads: {dead}"
