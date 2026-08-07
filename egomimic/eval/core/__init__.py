"""CORE evaluators (DESIGN.md §2 ``egomimic/eval/{core}``).

The shared eval infrastructure + the algo-agnostic evaluators every policy line
reuses, curated here in DESIGN.md step 8 (``git mv``, no behaviour change):

  * :class:`Eval` / :class:`EvalVideo`            — the ABC eval bases.
  * :class:`EvalList` / :class:`EvalVideoList`     — composite evaluators
    (``EvalListSideBySide`` is a back-compat alias of ``EvalVideoList``).
  * :class:`SimRolloutEval` / :class:`PackedSimEval` / :class:`HNetSimEval` /
    :class:`HPTSimEval`                            — closed-loop sim evaluators.
  * :class:`HNetEvalVideo`                         — the H-Net teacher-forced
    val overlay (the dense GT-vs-pred signal for the hourglass/BC line).
  * :class:`VAEReconEval`                          — VAE reconstruction val.
"""

from egomimic.eval.core.eval import Eval
from egomimic.eval.core.eval_video import EvalVideo
from egomimic.eval.core.eval_composite import (
    EvalList,
    EvalListSideBySide,
    EvalVideoList,
)
from egomimic.eval.core.eval_sim import (
    HNetSimEval,
    HPTSimEval,
    PackedSimEval,
    SimRolloutEval,
)
from egomimic.eval.core.eval_hnet import HNetEvalVideo
from egomimic.eval.core.eval_vae_recon import VAEReconEval

__all__ = [
    "Eval",
    "EvalVideo",
    "EvalList",
    "EvalVideoList",
    "EvalListSideBySide",
    "SimRolloutEval",
    "PackedSimEval",
    "HNetSimEval",
    "HPTSimEval",
    "HNetEvalVideo",
    "VAEReconEval",
]
