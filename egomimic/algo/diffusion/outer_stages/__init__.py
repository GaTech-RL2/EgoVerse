"""DFoT per-variant OUTER-STAGES (DESIGN.md §2
``egomimic/algo/diffusion/outer_stages/``).

The :class:`OuterStage` subclasses that wire the diffusion backbone + noise
process into each DFoT training variant (spatial / pixel / obs-action / video /
policy / decoupled). Relocated here from ``egomimic/algo/dfot`` in DESIGN.md
step 7 (``git mv``, no behaviour change); the base :class:`DFoTOuterStage` lives
in ``outer_stage.py`` and the rest subclass it.
"""

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
from egomimic.algo.diffusion.outer_stages.pixel_obs_action_decoupled_outer_stage import (  # noqa: E501
    PixelObsActionDecoupledDFoTOuterStage,
)
from egomimic.algo.diffusion.outer_stages.pixel_obs_action_policy_outer_stage import (  # noqa: E501
    PixelObsActionPolicyDFoTOuterStage,
)
from egomimic.algo.diffusion.outer_stages.pixel_obs_action_regress_outer_stage import (  # noqa: E501
    PixelObsActionRegressPolicyDFoTOuterStage,
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

__all__ = [
    "DFoTOuterStage",
    "make_dfot_ctx",
    "ImageSpatialDFoTOuterStage",
    "ObsActionImageDFoTOuterStage",
    "ObsActionDFoTOuterStage",
    "PixelObsActionDecoupledDFoTOuterStage",
    "PixelObsActionPolicyDFoTOuterStage",
    "PixelObsActionRegressPolicyDFoTOuterStage",
    "PixelSpatialDFoTOuterStage",
    "PixelVideoDFoTOuterStage",
    "SpatialObsActionPolicyDFoTOuterStage",
]
