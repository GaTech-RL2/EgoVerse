"""Repo-wide model-config preflight.

Every failure here is otherwise paid for on a training node AFTER the dataset
pull and the norm-stats phase — hours in, GPUs already allocated. Two real
cases from this sweep:

  * `act_seq: 16` against `action_horizon: 17`. CrossTransformer adds a
    (1, act_seq, D) positional table to the token sequence, so nine `append`
    layout arc configs died on their first forward pass.
  * an unexpected `rotation_radius` reaching a transform factory that had no
    such parameter, which killed 8 of 9 runs in one batch.

The static check runs over EVERY model config, because it is nearly free. The
forward-pass check runs over the PushShapes cotrain family — the configs this
repo actually sweeps — because instantiating all 38 would be slow without
telling us much: legacy configs reference embodiments and norm stats that no
longer exist, so failures there would be noise rather than signal.

A static check only catches mismatches someone already thought to look for.
Running the module is what catches the next one.
"""

from __future__ import annotations

import pathlib

import hydra
import pytest
import torch
from omegaconf import OmegaConf

REPO = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "egomimic/hydra_configs/model/bf"

ALL_CONFIGS = sorted(p.name for p in MODEL_DIR.glob("*.yaml"))
# PushShapes families whose stage graph is condition -> sampler/decoder and
# can therefore be driven from a synthetic condition without a dataset.
RUNNABLE = sorted(
    p.name for p in MODEL_DIR.glob("*.yaml")
    if p.name.startswith(("bf_cotrain11_", "bf_ctrlmode_"))
)


def _model(name: str):
    return OmegaConf.load(MODEL_DIR / name).robomimic_model


def _stage_list(cfg):
    return list(cfg.get("stages") or [])


def _denoisers(cfg):
    for stage in _stage_list(cfg):
        denoiser = stage.get("denoising_module")
        if denoiser is not None:
            yield stage, denoiser


def test_the_config_set_is_non_empty():
    assert ALL_CONFIGS and RUNNABLE


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_act_seq_equals_action_horizon(name):
    """CrossTransformer ADDS a (1, act_seq, D) table — it does not slice it.

    So act_seq must equal the horizon exactly. `act_seq < horizon` fails to
    broadcast; `act_seq > horizon` also fails, and would be a silently unused
    table even if it did not.
    """
    cfg = _model(name)
    if "action_horizon" not in cfg:
        pytest.skip(f"{name}: no action_horizon")
    horizon = int(cfg.action_horizon)
    checked = 0
    for _stage, denoiser in _denoisers(cfg):
        if "act_seq" not in denoiser:
            continue
        checked += 1
        assert int(denoiser.act_seq) == horizon, (
            f"{name}: act_seq={denoiser.act_seq} != action_horizon={horizon}. "
            "CrossTransformer adds a (1, act_seq, D) positional table to the "
            "token sequence; this dies on the first forward pass, after the "
            "dataset pull."
        )
    if not checked:
        pytest.skip(f"{name}: no act_seq to check")


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_sampler_horizon_agrees_across_stages(name):
    """GaussianLatentNoise and the sampler must agree on the horizon.

    They are configured independently, so a horizon change applied to one and
    not the other produces a shape error only at runtime.
    """
    cfg = _model(name)
    if "action_horizon" not in cfg:
        pytest.skip(f"{name}: no action_horizon")
    horizon = int(cfg.action_horizon)
    for stage in _stage_list(cfg):
        if "action_horizon" in stage:
            assert int(stage.action_horizon) == horizon, (
                f"{name}: {stage._target_.rsplit('.', 1)[-1]} declares "
                f"action_horizon={stage.action_horizon}, model declares {horizon}"
            )


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_latent_dim_agrees_between_noise_and_sampler(name):
    """The noise tensor feeds the sampler directly; a mismatch is a runtime
    shape error rather than a config error."""
    cfg = _model(name)
    dims = {
        stage._target_.rsplit(".", 1)[-1]: int(stage.latent_dim)
        for stage in _stage_list(cfg)
        if "latent_dim" in stage
    }
    if len(dims) < 2:
        pytest.skip(f"{name}: fewer than two latent_dim declarations")
    assert len(set(dims.values())) == 1, f"{name}: conflicting latent_dim {dims}"


@pytest.mark.parametrize("name", RUNNABLE)
def test_config_instantiates_and_runs_a_forward(name):
    """The backstop: build the real modules and push a batch through them."""
    cfg = _model(name)
    # action_dims lives on the sampler/decoder STAGE, not on robomimic_model.
    action_dims, cond_dim = None, None
    for raw in _stage_list(cfg):
        if "action_dims" in raw:
            action_dims = raw.action_dims
        if "condition_input_dim" in raw:
            cond_dim = int(raw.condition_input_dim)
    if cond_dim is None or action_dims is None:
        pytest.skip(f"{name}: no condition_input_dim/action_dims to build from")

    domain = str(next(iter(action_dims.keys())))
    # Skip the observation encoder: it needs real images, and what the arms
    # differ in is the head, which a synthetic condition can drive.
    head = [
        hydra.utils.instantiate(raw)
        for raw in _stage_list(cfg)
        if not raw._target_.endswith("FusedObsEncoder")
    ]

    horizon = int(cfg.action_horizon)
    action_dim = int(action_dims[domain])
    batch = {
        "condition": torch.randn(2, cond_dim),
        "embodiment": domain,
        "target": torch.randn(2, horizon, action_dim),
    }
    for stage in head:
        stage.eval()
        with torch.no_grad():
            batch = stage(batch)
    assert batch["pred_action"].shape == (2, horizon, action_dim), name
    assert torch.isfinite(batch["loss/native_action"]), name
