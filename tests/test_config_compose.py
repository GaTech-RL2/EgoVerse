"""Hydra config-composition invariants.

Distilled from the session's ``--cfg job`` compose sweeps. Composes each config
THROUGH the real parent ``train_zarr_cartesian`` defaults tree (the same path
``python -m egomimic.trainHydra model=<name> --cfg job`` exercises), so a broken
``_target_`` (e.g. the step-6 role-path move) or a missing default surfaces as a
test failure. Composition only RESOLVES the config (it does not instantiate the
torch model), so this runs fast on CPU or GPU.

The model checks retain the original BC-RNN/DFoT/VAE coverage.  The family
checks compose every data, evaluator, callback, trainer, logger, schematic,
visualization, and experiment choice through the real training root so nested
renames and ``@_here_`` package mistakes cannot land silently.
"""
from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Absolute path to egomimic/hydra_configs (this file lives in <repo>/tests/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "egomimic", "hydra_configs")
_PARENT = "train_zarr_cartesian"

BC_RNN_CONFIGS = [
    "bc_rnn/base",
    "bc_rnn/hnet",
    "bc_rnn/tx",
    "bc_rnn/tx_chunk8",
    "bc_rnn/tx_chunk8_q",
    "bc_rnn/tx_cos",
    "bc_rnn/tx_cos_lowlr",
]

DFOT_CONFIGS = [
    "dfot/pixel_video",
    "dfot/base",
    "dfot/image_spatial",
    "dfot/image_spatial_continuous",
    "dfot/image_spatial_cont_sigmoid",
    "dfot/image_spatial_policy",
    "dfot/obs_action",
    "dfot/obs_action_image",
    "dfot/obs_action_image_wm",
    "dfot/pixel",
    "dfot/pixel_decoupled",
    "dfot/pixel_policy",
    "dfot/pixel_regress",
]

VAE_CONFIGS = [
    "vae/base",
    "vae/v3",
    "vae/v4",
    "vae/v5",
    "vae/v6",
]


def _compose(model_name: str):
    # Fresh Hydra context per compose so the global singleton never leaks state
    # between parametrized cases.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name=_PARENT, overrides=[f"model={model_name}"])
    assert cfg.model is not None, f"{model_name}: composed cfg has no model node"
    return cfg


def _compose_overrides(*overrides: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name=_PARENT, overrides=list(overrides))


def _choices(group: str, *, exclude_dirs: tuple[str, ...] = ()) -> list[str]:
    root = os.path.join(_CONFIG_DIR, *group.split("/"))
    choices = []
    for dirpath, _, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        first = rel_dir.split(os.sep, 1)[0]
        if rel_dir != "." and first in exclude_dirs:
            continue
        for filename in filenames:
            if filename.endswith(".yaml"):
                rel = os.path.relpath(os.path.join(dirpath, filename), root)
                choices.append(os.path.splitext(rel)[0].replace(os.sep, "/"))
    return sorted(choices)


GROUP_CHOICES = [
    *[("data", name) for name in _choices("data")],
    *[("evaluator", name) for name in _choices("evaluator", exclude_dirs=("viz",))],
    *[("callbacks", name) for name in _choices("callbacks")],
    *[("trainer", name) for name in _choices("trainer")],
    *[("logger", name) for name in _choices("logger")],
]

VIZ_CHOICES = _choices("evaluator/viz")
EXPERIMENT_CHOICES = [
    name for name in _choices("experiment/indomain_c4") if name != "base"
]


@pytest.mark.parametrize("name", BC_RNN_CONFIGS)
def test_bc_rnn_config_composes(name):
    """MANDATORY invariant: all 7 BC-RNN configs compose with role-path targets."""
    cfg = _compose(name)
    # Every _target_ under the model tree must now point at a role home
    # (models.{stems,cores,heads}) — NOT the legacy bc_rnn_nets path.
    targets = _collect_targets(cfg.model)
    legacy = [t for t in targets if "models.bc_rnn_nets" in t]
    assert not legacy, f"{name}: still references legacy bc_rnn_nets: {legacy}"
    role = [
        t
        for t in targets
        if t.startswith("egomimic.models.stems.")
        or t.startswith("egomimic.models.cores.")
        or t.startswith("egomimic.models.heads.")
    ]
    assert role, f"{name}: no role-home _target_ found in composed model: {targets}"


@pytest.mark.parametrize("name", DFOT_CONFIGS)
def test_dfot_config_composes(name):
    _compose(name)


@pytest.mark.parametrize("name", VAE_CONFIGS)
def test_vae_config_composes(name):
    _compose(name)


@pytest.mark.parametrize("group,name", GROUP_CHOICES)
def test_config_group_choice_composes(group, name):
    _compose_overrides(f"{group}={name}")


@pytest.mark.parametrize("name", VIZ_CHOICES)
def test_viz_choice_composes(name):
    _compose_overrides(f"evaluator/viz@evaluator.viz_func={name}")


@pytest.mark.parametrize("name", EXPERIMENT_CHOICES)
def test_indomain_experiment_composes(name):
    _compose_overrides(f"+experiment=indomain_c4/{name}")


def test_large_config_groups_have_no_flat_variants():
    """Keep family variants nested; only a group-wide base may stay at root."""
    for group in ("data", "evaluator", "callbacks", "trainer", "logger"):
        root = os.path.join(_CONFIG_DIR, group)
        flat = sorted(
            filename
            for filename in os.listdir(root)
            if filename.endswith(".yaml") and filename != "base.yaml"
        )
        assert not flat, f"{group} has flat config variants: {flat}"


def _collect_targets(node):
    """Recursively gather every ``_target_`` string under a DictConfig/ListConfig."""
    from omegaconf import DictConfig, ListConfig

    out = []
    if isinstance(node, DictConfig):
        for k, v in node.items():
            if k == "_target_" and isinstance(v, str):
                out.append(v)
            else:
                out.extend(_collect_targets(v))
    elif isinstance(node, ListConfig):
        for v in node:
            out.extend(_collect_targets(v))
    return out
