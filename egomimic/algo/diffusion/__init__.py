"""DFoT (Diffusion Forcing Transformer) ALGO package (DESIGN.md §2
``egomimic/algo/diffusion/``).

Role home for the *algo* half of the diffusion stack — the :class:`DFoT` algo,
its per-variant outer-stages, and the VAE pre-train algo. Relocated here from
``egomimic/algo/dfot`` (+ ``egomimic/algo/vae``) in DESIGN.md step 7 via
``git mv`` (no behaviour change); the *model* half (backbones / diffusion
processes / embeddings / sampling) lives at ``egomimic.models.diffusion``.

  * :class:`DFoT`                      — the diffusion-forcing algo.
  * ``outer_stages/``                  — per-variant OuterStage subclasses
                                         (spatial / pixel / obs-action / ...).
  * :class:`VAE` + ``load_pretrained_vae`` (``vae_algo``) — image VAE pre-train.

The legacy ``egomimic.algo.dfot`` / ``egomimic.algo.vae`` facade shims (and the
yaml ``_target_``s that routed through them) were deleted at the DESIGN.md
step-13 final flip; every consumer now imports from ``egomimic.algo.diffusion``
/ ``egomimic.models.diffusion`` directly.
"""

from egomimic.algo.diffusion.algo import DFoT
from egomimic.algo.diffusion.outer_stages.outer_stage import (
    DFoTOuterStage,
    make_dfot_ctx,
)
from egomimic.algo.diffusion.outer_stages.image_spatial_outer_stage import (
    ImageSpatialDFoTOuterStage,
)
from egomimic.algo.diffusion.outer_stages.obs_action_image_outer_stage import (
    ObsActionImageDFoTOuterStage,
)
from egomimic.algo.diffusion.outer_stages.obs_action_outer_stage import (
    ObsActionDFoTOuterStage,
)
from egomimic.algo.diffusion.outer_stages.pixel_obs_action_outer_stage import (  # noqa: E501
    PixelObsActionDFoTOuterStage,
)
from egomimic.algo.diffusion.outer_stages.pixel_spatial_outer_stage import (
    PixelSpatialDFoTOuterStage,
)
from egomimic.algo.diffusion.outer_stages.pixel_video_outer_stage import (
    PixelVideoDFoTOuterStage,
)
from egomimic.algo.diffusion.outer_stages.spatial_obs_action_policy_outer_stage import (  # noqa: E501
    SpatialObsActionPolicyDFoTOuterStage,
)
from egomimic.algo.diffusion.vae_algo import VAE, load_pretrained_vae

# Convenience re-exports of the model-half pieces some legacy code imports via
# the algo namespace (kept here so ``from egomimic.algo.diffusion import
# DFoTBackbone`` mirrors the model-half surface).
from egomimic.models.diffusion import (
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
    "DFoTOuterStage",
    "make_dfot_ctx",
    "ImageSpatialDFoTOuterStage",
    "ObsActionImageDFoTOuterStage",
    "ObsActionDFoTOuterStage",
    "PixelObsActionDFoTOuterStage",
    "PixelSpatialDFoTOuterStage",
    "PixelVideoDFoTOuterStage",
    "SpatialObsActionPolicyDFoTOuterStage",
    "VAE",
    "load_pretrained_vae",
    # model-half re-exports
    "DFoTBackbone",
    "DFoTDiT3DBackbone",
    "DFoTSpatialBackbone",
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
