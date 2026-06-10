"""Review-pass regression tests: the D03 ``image_range`` seam must cover the
CLOSED-LOOP pixel inference context paths in ``algo.py`` — the three
``_inference_step_pixel_{policy,regress,decoupled}`` controllers — not just
the training/rollout/eval seams owned by the outer stage.

Two contracts per controller:

  1. Bit-compat at the default ``image_range="01"``: the stored context frame
     is byte-identical to the historic ``uint8 / 255`` extraction (the seam
     is an identity pass-through, same tensor values, no extra ops).
  2. ``image_range="pm1"``: the stored context frame is ``(x/255)*2 - 1`` —
     the same model range training pins via ``_extract_images``, so the
     closed-loop context is no longer range-inconsistent under the parity
     recipe (``dfot_pushshapes_pixel_policy_parity.yaml``).

Each test drives the REAL controller end-to-end on CPU (real outer stage,
real ``DiscreteDiffusion``, shape-preserving stub backbone) and also sanity-
checks the returned action. Pure torch, tiny shapes, no data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from egomimic.algo.diffusion.algo import DFoT
from egomimic.algo.diffusion.outer_stages.pixel_obs_action_outer_stage import (
    PixelObsActionDFoTOuterStage,
)
from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

A = 2
C_IMG = 3
HW = 16
TIMESTEPS = 8
IMG_KEY = "front_img_1"
AC_KEY = "actions"
EMB_ID = get_embodiment_id("pushshapes_sim")


class _TinyBackbone(nn.Module):
    """Shape-preserving stand-in for the DiT3D; the parameter feeds the
    controller's device probe. Also answers the decoupled dual-stream call."""

    action_token_dim = A  # decoupled-mode ctor check

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, x, k, external_cond=None, action=None,
                action_noise_levels=None, **_):
        if action is not None:
            return x * 0.5, action * 0.5
        return x * 0.5


class _IdentityNormStats:
    def normalize(self, data, emb_id):
        return data

    def unnormalize(self, data, emb_id):
        return data


def _make_algo(pixel_mode: str, image_range: str) -> DFoT:
    stage = PixelObsActionDFoTOuterStage(
        action_dim=A,
        cond_encoder=None,
        backbone=_TinyBackbone(),
        diffusion=DiscreteDiffusion(action_dim=A, timesteps=TIMESTEPS),
        image_key=IMG_KEY,
        image_channels=C_IMG,
        image_size=HW,
        frame_sampling="full",
        pixel_mode=pixel_mode,
        action_channels=A if pixel_mode == "policy" else None,
        head_width=16,
        image_range=image_range,
    )
    algo = object.__new__(DFoT)
    algo.nets = {"outer_stage": stage}
    algo.norm_stats = _IdentityNormStats()
    algo.ac_keys = {"pushshapes_sim": AC_KEY}
    algo.action_dim = A
    algo.sampler_n_steps = 2
    algo.sp_n_context = 2
    algo.sp_commit = 1
    algo.sp_n_samples = 1
    return algo


def _obs(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {
        IMG_KEY: torch.randint(0, 256, (C_IMG, HW, HW), dtype=torch.uint8),
        "state_agent_obj": torch.tensor([0.25, -0.5, 0.0, 0.0]),
    }


_CONTROLLERS = [
    ("policy", "_inference_step_pixel_policy", "_pp_rgb"),
    ("regress", "_inference_step_pixel_regress", "_pr_rgb"),
    ("decoupled", "_inference_step_pixel_decoupled", "_pd_rgb"),
]


@pytest.mark.parametrize("mode,method,buf", _CONTROLLERS)
def test_default_01_context_storage_bit_compat(mode, method, buf):
    algo = _make_algo(mode, "01")
    obs = _obs()
    torch.manual_seed(42)
    action = getattr(algo, method)(obs, 0, EMB_ID)
    stored = getattr(algo, buf)[0]
    # identity seam: exactly the historic uint8/255 extraction
    assert torch.equal(stored, obs[IMG_KEY].float() / 255.0)
    assert action.shape == (A,)
    assert action.dtype == np.float32


@pytest.mark.parametrize("mode,method,buf", _CONTROLLERS)
def test_pm1_context_storage_is_model_range(mode, method, buf):
    algo = _make_algo(mode, "pm1")
    obs = _obs()
    torch.manual_seed(42)
    action = getattr(algo, method)(obs, 0, EMB_ID)
    stored = getattr(algo, buf)[0]
    expected = (obs[IMG_KEY].float() / 255.0) * 2.0 - 1.0
    assert torch.equal(stored, expected)
    assert stored.min() >= -1.0 and stored.max() <= 1.0
    assert action.shape == (A,)
    assert action.dtype == np.float32


def test_01_vs_pm1_differ_only_via_range_map():
    """Same RNG seed: the '01' and 'pm1' controllers consume identical
    randomness; only the pinned context (and downstream sample) differs."""
    obs = _obs()
    algo01 = _make_algo("policy", "01")
    torch.manual_seed(7)
    algo01._inference_step_pixel_policy(obs, 0, EMB_ID)
    algopm = _make_algo("policy", "pm1")
    torch.manual_seed(7)
    algopm._inference_step_pixel_policy(obs, 0, EMB_ID)
    s01, spm = algo01._pp_rgb[0], algopm._pp_rgb[0]
    assert torch.equal(spm, s01 * 2.0 - 1.0)
