"""Backward-compat FACADE — ``egomimic.algo.dfot`` was SPLIT in DESIGN.md step 7:

  * the *model* half (backbones / diffusion processes / embeddings / sampling)
    moved to ``egomimic.models.diffusion`` (``models/diffusion/{backbones,
    diffusion}`` + ``embeddings`` + ``sampling``);
  * the *algo* half (``DFoT`` + per-variant outer-stages + the VAE pre-train
    algo) moved to ``egomimic.algo.diffusion`` (``algo/diffusion/{algo,
    outer_stages,vae_algo}``).

This package is kept ALIVE as a thin FACADE so every legacy import path the repo
/ external scratch / yaml ``_target_``s use keeps working unchanged until the
final flip (DESIGN.md step 13):

  * ``from egomimic.algo.dfot import DFoT, DFoTBackbone, ...``                 (names)
  * ``from egomimic.algo.dfot.<sub> import ...``                              (submodules)
  * ``_target_: egomimic.algo.dfot.DFoT``                                     (yaml, top-level)
  * ``_target_: egomimic.algo.dfot.outer_stage.DFoTOuterStage``              (yaml, submodule)
  * ``_target_: egomimic.algo.dfot.continuous_diffusion.ContinuousDiffusion`` (yaml, submodule)
  * ``_target_: egomimic.algo.dfot.discrete_diffusion.DiscreteDiffusion``     (yaml, submodule)

Mechanism (identical to the ``hnet_nets`` / ``bc_rnn_nets`` collapse shims from
DESIGN.md steps 3/6): each legacy ``egomimic.algo.dfot.<sub>`` dotted path is
registered in ``sys.modules`` pointing at the REAL relocated module (under
``egomimic.models.diffusion`` or ``egomimic.algo.diffusion.outer_stages``), so
``from egomimic.algo.dfot.<sub> import X`` and a yaml
``_target_: egomimic.algo.dfot.<sub>.X`` both resolve to the SAME module object
(identical symbols, including private ones). No code is duplicated.

DO NOT add logic here — it is a pure forwarder. When DESIGN.md step 13 flips
every consumer to ``egomimic.models.diffusion`` / ``egomimic.algo.diffusion``
directly, this whole package is deleted.
"""

import importlib as _importlib
import sys as _sys

# Legacy ``algo.dfot.<sub>``  ->  its real relocated module. Covers every
# submodule path the repo / yaml _target_s / eval code import by dotted path.
_SUBMODULE_HOMES = {
    # --- model half -> egomimic.models.diffusion ---
    "backbone": "egomimic.models.diffusion.backbones.backbone",
    "dit3d_backbone": "egomimic.models.diffusion.backbones.dit3d_backbone",
    "spatial_backbone": "egomimic.models.diffusion.backbones.spatial_backbone",
    "continuous_diffusion": "egomimic.models.diffusion.diffusion.continuous_diffusion",  # noqa: E501
    "discrete_diffusion": "egomimic.models.diffusion.diffusion.discrete_diffusion",  # noqa: E501
    "noise_schedule": "egomimic.models.diffusion.diffusion.noise_schedule",
    "embeddings": "egomimic.models.diffusion.embeddings",
    "sampling": "egomimic.models.diffusion.sampling",
    # --- algo half -> egomimic.algo.diffusion ---
    "algo": "egomimic.algo.diffusion.algo",
    "outer_stage": "egomimic.algo.diffusion.outer_stages.outer_stage",
    "image_spatial_outer_stage": "egomimic.algo.diffusion.outer_stages.image_spatial_outer_stage",  # noqa: E501
    "obs_action_image_outer_stage": "egomimic.algo.diffusion.outer_stages.obs_action_image_outer_stage",  # noqa: E501
    "obs_action_outer_stage": "egomimic.algo.diffusion.outer_stages.obs_action_outer_stage",  # noqa: E501
    "pixel_obs_action_decoupled_outer_stage": "egomimic.algo.diffusion.outer_stages.pixel_obs_action_decoupled_outer_stage",  # noqa: E501
    "pixel_obs_action_policy_outer_stage": "egomimic.algo.diffusion.outer_stages.pixel_obs_action_policy_outer_stage",  # noqa: E501
    "pixel_obs_action_regress_outer_stage": "egomimic.algo.diffusion.outer_stages.pixel_obs_action_regress_outer_stage",  # noqa: E501
    "pixel_spatial_outer_stage": "egomimic.algo.diffusion.outer_stages.pixel_spatial_outer_stage",  # noqa: E501
    "pixel_video_outer_stage": "egomimic.algo.diffusion.outer_stages.pixel_video_outer_stage",  # noqa: E501
    "spatial_obs_action_policy_outer_stage": "egomimic.algo.diffusion.outer_stages.spatial_obs_action_policy_outer_stage",  # noqa: E501
}

for _legacy, _home in _SUBMODULE_HOMES.items():
    _mod = _importlib.import_module(_home)
    _sys.modules[f"{__name__}.{_legacy}"] = _mod

# Top-level name re-exports (mirror the pre-split ``algo.dfot.__init__``).
from egomimic.algo.diffusion import (  # noqa: E402
    DFoT,
    DFoTOuterStage,
    ImageSpatialDFoTOuterStage,
    ObsActionDFoTOuterStage,
    ObsActionImageDFoTOuterStage,
    PixelObsActionDecoupledDFoTOuterStage,
    PixelObsActionPolicyDFoTOuterStage,
    PixelObsActionRegressPolicyDFoTOuterStage,
    PixelSpatialDFoTOuterStage,
    PixelVideoDFoTOuterStage,
    SpatialObsActionPolicyDFoTOuterStage,
    make_dfot_ctx,
)
from egomimic.models.diffusion import (  # noqa: E402
    ContinuousDiffusion,
    DFoTBackbone,
    DFoTDiT3DBackbone,
    DFoTSpatialBackbone,
    DiscreteDiffusion,
    causal_ar_schedule,
    ddim_sample,
    ddpm_sample,
    sample,
    sample_step,
    staircase_ar_schedule,
    vanilla_schedule,
)

__all__ = [
    "DFoT",
    "DFoTBackbone",
    "DFoTDiT3DBackbone",
    "DFoTOuterStage",
    "DFoTSpatialBackbone",
    "ImageSpatialDFoTOuterStage",
    "ObsActionDFoTOuterStage",
    "ObsActionImageDFoTOuterStage",
    "PixelObsActionDecoupledDFoTOuterStage",
    "PixelObsActionPolicyDFoTOuterStage",
    "PixelObsActionRegressPolicyDFoTOuterStage",
    "PixelSpatialDFoTOuterStage",
    "PixelVideoDFoTOuterStage",
    "SpatialObsActionPolicyDFoTOuterStage",
    "make_dfot_ctx",
    "ContinuousDiffusion",
    "DiscreteDiffusion",
    "sample_step",
    "sample",
    "vanilla_schedule",
    "causal_ar_schedule",
    "staircase_ar_schedule",
    "ddim_sample",
    "ddpm_sample",
]

del _sys, _importlib, _legacy, _home, _mod, _SUBMODULE_HOMES
