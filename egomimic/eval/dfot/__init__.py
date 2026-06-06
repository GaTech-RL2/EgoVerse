"""DFoT closed-loop / rollout evaluators (DESIGN.md §2 ``egomimic/eval/{dfot}``).

The DFoT self-rollout (world-model) + policy-action + video-rollout evaluators,
curated here in DESIGN.md step 8 (``git mv``, no behaviour change):

  * :class:`DFoTSelfRolloutEval`        — joint obs+action world-model rollout.
  * :class:`DFoTVideoRolloutEval` / :class:`DFoTPixelVideoRolloutEval` /
    :class:`DFoTSpatialVideoRolloutEval` — video self-rollout variants.
  * :class:`DFoTPolicyActionEval` / :class:`DFoTPolicyRecedingHorizonEval` —
    2D-policy action prediction (whole-horizon vs receding-horizon).
  * :class:`DFoTBundleAnchoredEval`     — anchored clean-history bundle rollout.
"""

from egomimic.eval.dfot.eval_dfot_self_rollout import DFoTSelfRolloutEval
from egomimic.eval.dfot.eval_dfot_video_rollout import DFoTVideoRolloutEval
from egomimic.eval.dfot.eval_dfot_pixel_video_rollout import (
    DFoTPixelVideoRolloutEval,
)
from egomimic.eval.dfot.eval_dfot_spatial_video_rollout import (
    DFoTSpatialVideoRolloutEval,
)
from egomimic.eval.dfot.eval_dfot_policy_action import DFoTPolicyActionEval
from egomimic.eval.dfot.eval_dfot_policy_receding_horizon import (
    DFoTPolicyRecedingHorizonEval,
)
from egomimic.eval.dfot.eval_dfot_bundle_anchored import DFoTBundleAnchoredEval

__all__ = [
    "DFoTSelfRolloutEval",
    "DFoTVideoRolloutEval",
    "DFoTPixelVideoRolloutEval",
    "DFoTSpatialVideoRolloutEval",
    "DFoTPolicyActionEval",
    "DFoTPolicyRecedingHorizonEval",
    "DFoTBundleAnchoredEval",
]
