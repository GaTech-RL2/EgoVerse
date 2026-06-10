"""Parity-recipe yaml wiring (Port Parity Checklist — Wire agent).

Guards the two true bug repairs in ``dfot_pushshapes_pixel_policy.yaml``
(D01 inference_mode, D05 eta_min) and the full upstream-validated recipe in
``dfot_pushshapes_pixel_policy_parity.yaml``: every parity key must compose
through the real ``train_zarr_cartesian`` defaults tree AND name a real ctor
param of its ``_target_`` (a typo'd knob would otherwise be silently dropped
into an unexpected-kwarg crash at training time, or worse — accepted and
ignored). Also locks the fixed-/512 minmax norm-stats JSON to the schema
``MultiDataset.infer_norm_from_dataset`` / ``_check_bounds`` consume.

Pure compose + introspection + torch-on-CPU; no data, no instantiation of
the backbone.
"""
from __future__ import annotations

import inspect
import json
import os

import pytest
import torch
from hydra import compose, initialize_config_dir

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "egomimic", "hydra_configs")
_PARENT = "train_zarr_cartesian"
_NORM_JSON = os.path.join(
    _CONFIG_DIR, "data", "norm_stats_pushshapes_fixed512.json"
)


def _compose(model_name: str, extra: list[str] | None = None):
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base="1.3"):
        return compose(
            config_name=_PARENT,
            overrides=[f"model={model_name}", "data=tsimulation"]
            + (extra or []),
        )


# ---------------------------------------------------------------------------
# Base yaml — the two designated true bug fixes, and nothing else semantic.
# ---------------------------------------------------------------------------


def test_base_yaml_d01_pixel_policy_mode():
    cfg = _compose("dfot_pushshapes_pixel_policy")
    assert cfg.model.robomimic_model.inference_mode == "pixel_policy"


def test_base_yaml_d05_eta_min_left_family_wide():
    # rev-2 audit: eta_min == lr is a FAMILY-WIDE pixel-config setting
    # (all 5 configs), deliberate-looking but unvalidated — NOT a typo.
    # The base config keeps it for reproducibility; the validated cosine
    # (eta_min 5e-6 < lr) lives only in the parity yaml.
    cfg = _compose("dfot_pushshapes_pixel_policy")
    assert cfg.model.scheduler.eta_min == pytest.approx(2.0e-4)
    assert cfg.model.scheduler.eta_min == pytest.approx(cfg.model.optimizer.lr)


def test_base_yaml_training_semantics_untouched():
    """Everything except the two bug fixes keeps the original values."""
    cfg = _compose("dfot_pushshapes_pixel_policy")
    m = cfg.model.robomimic_model
    assert m.sampler_n_steps == 50
    assert m.outer_stage.diffusion.objective == "pred_v"
    assert m.outer_stage.diffusion.loss_weighting_strategy == "fused_min_snr"
    assert "image_range" not in m.outer_stage  # default "01" via ctor
    assert cfg.model.optimizer.lr == pytest.approx(2.0e-4)
    assert cfg.model.optimizer.betas == [0.9, 0.99]


# ---------------------------------------------------------------------------
# Parity yaml — every D-delta present with the validated value.
# ---------------------------------------------------------------------------


def test_parity_yaml_values():
    cfg = _compose("dfot_pushshapes_pixel_policy_parity")
    m = cfg.model.robomimic_model
    assert m.inference_mode == "pixel_policy"  # D01
    assert m.sampler_n_steps == 25  # D20
    assert m.sp_n_context == 8  # D13: n_ctx + sp_commit(1) == trained T=9
    outer = m.outer_stage
    assert outer.image_range == "pm1"  # D03a
    assert outer.image_loss_weight == 0.5  # D16
    assert outer.action_loss_weight_group == 0.5  # D16
    diff = outer.diffusion
    assert diff.objective == "pred_x0"  # D04
    assert diff.loss_weighting_strategy == "constant"  # D04
    assert diff.constant_weight == 2.5  # D04 (snr_clip * 0.5 upstream)
    assert diff.max_beta == pytest.approx(0.999)  # D15
    assert cfg.model.optimizer.lr == pytest.approx(5.0e-5)  # D05
    assert cfg.model.optimizer.weight_decay == pytest.approx(1.0e-4)  # D09
    assert list(cfg.model.optimizer.betas) == [0.9, 0.999]  # D09
    assert cfg.model.scheduler.warmup_steps == 2500  # D05/D23
    assert cfg.model.scheduler.eta_min == pytest.approx(5.0e-6)  # D05


def test_parity_companion_overrides_compose():
    """The documented out-of-model-package overrides must be accepted."""
    cfg = _compose(
        "dfot_pushshapes_pixel_policy_parity",
        extra=[
            "trainer.gradient_clip_val=10.0",
            "norm_stats.norm_mode=minmax",
            f"norm_stats.precomputed_norm_path={_NORM_JSON}",
            "reject_outliers=false",
        ],
    )
    assert cfg.trainer.gradient_clip_val == 10.0  # D09
    assert cfg.norm_stats.norm_mode == "minmax"  # D03b
    assert cfg.reject_outliers is False  # D30


def test_trainer_default_grad_clip_is_null():
    """D09 knob exists but defaults to disabled (bit-compat rule)."""
    cfg = _compose("dfot_pushshapes_pixel_policy")
    assert cfg.trainer.gradient_clip_val is None


# ---------------------------------------------------------------------------
# Every yaml key names a real ctor param of its _target_ (no invented knobs).
# ---------------------------------------------------------------------------


def _ctor_params(target: str) -> tuple[set, bool]:
    module, _, name = target.rpartition(".")
    obj = getattr(__import__(module, fromlist=[name]), name)
    fn = obj.__init__ if inspect.isclass(obj) else obj
    sig = inspect.signature(fn)
    return set(sig.parameters), any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )


@pytest.mark.parametrize(
    "node_path",
    [
        "model.robomimic_model",
        "model.robomimic_model.outer_stage",
        "model.robomimic_model.outer_stage.diffusion",
        "model.scheduler",
    ],
)
def test_parity_keys_exist_on_targets(node_path):
    from omegaconf import OmegaConf

    cfg = _compose("dfot_pushshapes_pixel_policy_parity")
    node = OmegaConf.select(cfg, node_path)
    params, has_var_kw = _ctor_params(node["_target_"])
    yaml_keys = {k for k in node.keys() if not k.startswith("_")}
    unknown = yaml_keys - params
    # **kwargs ctors (DFoT, outer stage) still must expose the parity knobs
    # we rely on as NAMED params somewhere in the MRO-visible signature; the
    # critical knobs are asserted explicitly here so a silent **kwargs
    # swallow cannot hide a typo.
    critical = {
        "model.robomimic_model": {"sp_n_context", "inference_mode",
                                  "sampler_n_steps"},
        "model.robomimic_model.outer_stage.diffusion": {
            "objective", "loss_weighting_strategy", "constant_weight",
            "max_beta",
        },
        "model.scheduler": {"warmup_steps", "eta_min", "max_steps"},
    }.get(node_path, set())
    assert critical <= params
    if not has_var_kw:
        assert not unknown, f"unknown keys for {node['_target_']}: {unknown}"


def test_parity_outer_stage_knobs_reach_base_ctor():
    """Outer-stage parity knobs flow through **kwargs to the base ctor."""
    from egomimic.algo.diffusion.outer_stages.pixel_spatial_outer_stage import (
        PixelSpatialDFoTOuterStage,
    )

    params, _ = _ctor_params(
        f"{PixelSpatialDFoTOuterStage.__module__}.PixelSpatialDFoTOuterStage"
    )
    assert {"image_range", "image_loss_weight", "action_loss_weight_group"} \
        <= params


# ---------------------------------------------------------------------------
# Fixed-minmax norm-stats JSON (D03b).
# ---------------------------------------------------------------------------


def test_norm_json_schema_and_math():
    with open(_NORM_JSON) as f:
        payload = json.load(f)
    # exact keys infer_norm_from_dataset / _check_bounds consume
    stats = payload["stats"]["15"]["actions"]  # 15 == PUSHSHAPES_SIM
    assert {"min", "max", "quantile_1", "quantile_99"} <= set(stats)
    assert stats["min"] == [0.0, 0.0]
    assert stats["max"] == [512.0, 512.0]
    # minmax branch: 2*((x-min)/(max-min+1e-6))-1 == upstream /512 -> [-1,1]
    x = torch.tensor([0.0, 256.0, 512.0])
    mn = torch.tensor(stats["min"][0])
    mx = torch.tensor(stats["max"][0])
    normed = 2.0 * ((x - mn) / (mx - mn + 1e-6)) - 1.0
    assert torch.allclose(
        normed, torch.tensor([-1.0, 0.0, 1.0]), atol=1e-5
    )
    # bounds check passes any in-range action (no silent sample drops)
    assert stats["quantile_1"] == stats["min"]
    assert stats["quantile_99"] == stats["max"]


def test_norm_json_embodiment_id_matches_enum():
    from egomimic.rldb.embodiment.embodiment import (
        EMBODIMENT,
        get_embodiment_id,
    )

    with open(_NORM_JSON) as f:
        payload = json.load(f)
    assert str(EMBODIMENT.PUSHSHAPES_SIM.value) in payload["stats"]
    # the id the algo resolves for the yaml's domain name
    assert str(get_embodiment_id("pushshapes_sim")) in payload["stats"]
