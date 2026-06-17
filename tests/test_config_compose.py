"""Hydra config-composition invariant (DESIGN.md amendment 6.5 + step-6/§5
invariant: "the 7 BC-RNN configs compose; DFoT/zoo import").

Distilled from the session's ``--cfg job`` compose sweeps. Composes each config
THROUGH the real parent ``train_zarr_cartesian`` defaults tree (the same path
``python -m egomimic.trainHydra model=<name> --cfg job`` exercises), so a broken
``_target_`` (e.g. the step-6 role-path move) or a missing default surfaces as a
test failure. Composition only RESOLVES the config (it does not instantiate the
torch model), so this runs fast on CPU or GPU.

The 7 BC-RNN configs are the MANDATORY invariant. The DFoT/VAE configs are
included so the relocation is shown not to have disturbed the wider model tree
(DESIGN.md "DFoT/zoo import" half of the invariant).
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
    "bc_rnn_pushshapes_paperexact",
    "bc_rnn_pushshapes_paperexact_hnet",
    "bc_rnn_pushshapes_paperexact_tx",
    "bc_rnn_pushshapes_paperexact_tx_chunk8",
    "bc_rnn_pushshapes_paperexact_tx_cos",
    "bc_rnn_pushshapes_paperexact_tx_cos_lowlr",
]

DFOT_CONFIGS = [
    "dfot_pixel_video",
    "dfot_pushshapes",
    "dfot_pushshapes_image_spatial",
    "dfot_pushshapes_image_spatial_continuous",
    "dfot_pushshapes_image_spatial_cont_sigmoid",
    "dfot_pushshapes_image_spatial_policy",
    "dfot_pushshapes_obs_action",
    "dfot_pushshapes_obs_action_image",
    "dfot_pushshapes_obs_action_image_wm",
    "dfot_pushshapes_pixel",
    "dfot_pushshapes_pixel_decoupled",
    "dfot_pushshapes_pixel_policy",
    "dfot_pushshapes_pixel_regress",
]

VAE_CONFIGS = [
    "vae_pushshapes",
    "vae_pushshapes_v3",
    "vae_pushshapes_v4",
    "vae_pushshapes_v5",
    "vae_pushshapes_v6",
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
