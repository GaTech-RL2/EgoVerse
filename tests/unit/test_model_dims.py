"""Per-embodiment widths are declared once under ``robomimic_model.dims`` and every
stem/head width in an HPT model config is an interpolation of them (Phase 3).

Checks, run on every ``model/*.yaml`` whose ``robomimic_model`` is HPT: the resolved
``dims`` are the numbers the yamls held before the refactor; every width leaf is
literally the expected ``${model.robomimic_model.dims...}`` string (a leftover
literal or a proprio/action mix-up cannot hide behind equal values); and
overriding each ``dims`` value with a distinct number moves every width bound to
it, and every value reaches at least one width (none is declared but unread).
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

COTRAIN = {"eva_bimanual": (14, 14), "human_bimanual": (12, 12)}
# One shared head: human actions are zero-padded to its width.
SHARED = (14, {"eva_bimanual": (14, 14), "human_bimanual": (12, 14)})

# {config: (action_width or None, {emb: (proprio, action)})}; None = no shared
# head, so no action_width key.
EXPECTED = {
    "hpt_bc_flow_eva": (None, {"eva_bimanual": (20, 20)}),
    # Human data defaults to the 144-D wrist-frame hand-keypoint action.
    "hpt_bc_flow_aria": (None, {"human_bimanual": (144, 144)}),
    "hpt_bc_flow_human": (None, {"human_bimanual": (144, 144)}),
    "hpt_bc_flow_mecka": (None, {"human_bimanual": (144, 144)}),
    "hpt_bc_flow_scale": (None, {"human_bimanual": (144, 144)}),
    "hpt_bc_flow_human_cartesian": (None, {"human_bimanual": (12, 12)}),
    "hpt_bc_keypoints_base": (None, {"human_bimanual": (144, 144)}),
    "hpt_bc_keypoints_wrist_300M": (None, {"human_bimanual": (144, 144)}),
    "hpt_bc_mecka_6d_300M": (None, {"human_bimanual": (20, 18)}),
    "hpt_bc_pickplace_qwen_pertoken": (None, {"eva_bimanual": (14, 14)}),
    "hpt_bc_pickplace_qwen_pooled": (None, {"eva_bimanual": (14, 14)}),
    "hpt_cotrain_enc_dec_base": (None, COTRAIN),
    # Separate heads: the human head follows the keypoint default.
    "hpt_cotrain_flow_seperate_head": (
        None,
        {"eva_bimanual": (14, 14), "human_bimanual": (144, 144)},
    ),
    "hpt_cotrain_flow_shared_head": SHARED,
    "hpt_cotrain_mecka_flow_shared_head": SHARED,
    "hpt_cotrain_scale_flow_shared_head": SHARED,
}


def _model(compose_resolve, name, extra=(), keep_hydra=False):
    cfg = compose_resolve(
        "train_zarr_cartesian", [f"model={name}", *extra], keep_hydra=keep_hydra
    )
    return cfg.model.robomimic_model


def _width_leaves(rm):
    """Yield (path, value, dims_key) for every width leaf of a resolved or
    unresolved robomimic_model container; dims_key is the ``dims`` entry the leaf
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


def _dims_keys(name):
    action_width, per_emb = EXPECTED[name]
    keys = [f"{emb}.{kind}" for emb in per_emb for kind in ("proprio", "action")]
    if action_width is not None:
        keys.append("action_width")
    return keys


def test_every_hpt_config_has_golden_dims():
    assert HPT_CONFIGS and set(HPT_CONFIGS) == set(EXPECTED)


@pytest.mark.parametrize("name", HPT_CONFIGS)
def test_dims_block_holds_golden_values(name, compose_resolve):
    action_width, per_emb = EXPECTED[name]
    want = {emb: {"proprio": p, "action": a} for emb, (p, a) in per_emb.items()}
    if action_width is not None:
        want["action_width"] = action_width
    assert OmegaConf.to_container(_model(compose_resolve, name).dims) == want


@pytest.mark.parametrize("name", HPT_CONFIGS)
def test_every_width_leaf_is_the_expected_interpolation(name, compose_resolve):
    rm = _model(compose_resolve, name, keep_hydra=True)
    raw = OmegaConf.to_container(rm, resolve=False)
    leaves = list(_width_leaves(raw))
    assert leaves, "no width leaves found"
    for path, value, key in leaves:
        assert value == f"${{model.robomimic_model.dims.{key}}}", path


@pytest.mark.parametrize("name", HPT_CONFIGS)
def test_every_dims_value_moves_its_widths(name, compose_resolve):
    distinct = {key: 101 + i for i, key in enumerate(_dims_keys(name))}
    overrides = [f"model.robomimic_model.dims.{k}={v}" for k, v in distinct.items()]
    rm = _model(compose_resolve, name, overrides)
    leaves = list(_width_leaves(OmegaConf.to_container(rm)))
    for path, value, key in leaves:
        assert value == distinct[key], path
    seen = {value for _path, value, _key in leaves}
    dead = [key for key, value in distinct.items() if value not in seen]
    assert not dead, f"dims values no width reads: {dead}"
