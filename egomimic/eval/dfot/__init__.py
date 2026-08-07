"""DFoT closed-loop / rollout evaluators (DESIGN.md §2 ``egomimic/eval/{dfot}``).

The DFoT self-rollout (world-model) + policy-action + video-rollout evaluators,
curated here in DESIGN.md step 8 (``git mv``, no behaviour change):

  * :class:`DFoTSelfRolloutEval`        — joint obs+action world-model rollout.
  * :class:`DFoTVideoRolloutEval` — family-agnostic video self-rollout
    (COMBINE A): drives the obs+action+image / image-spatial / pixel families
    via each outer stage's ``rollout_video_episode`` hook.
    :class:`DFoTPixelVideoRolloutEval` / :class:`DFoTSpatialVideoRolloutEval`
    are compat aliases of it (the per-family modules were collapsed in
    COMBINE A).
  * :class:`DFoTPolicyActionEval` / :class:`DFoTPolicyRecedingHorizonEval` —
    2D-policy action prediction (whole-horizon vs receding-horizon). COMBINE B
    merged the two into ``eval_dfot_policy`` (they share ``_rollout``).
  * :class:`DFoTBundleAnchoredEval`     — anchored clean-history bundle rollout.
"""

from egomimic.eval.dfot.eval_dfot_self_rollout import DFoTSelfRolloutEval
from egomimic.eval.dfot.eval_dfot_video_rollout import (
    DFoTVideoRolloutEval,
    DFoTPixelVideoRolloutEval,
    DFoTSpatialVideoRolloutEval,
)
from egomimic.eval.dfot.eval_dfot_policy import (
    DFoTPolicyActionEval,
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
