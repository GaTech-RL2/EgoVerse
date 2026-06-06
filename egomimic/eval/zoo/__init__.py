"""ZOO evaluators (DESIGN.md §2 ``egomimic/eval/{zoo}``).

The baseline-policy (ACT / HPT / PI) video evaluators that pair with the
``egomimic.algo.zoo`` algos — curated here in DESIGN.md step 8 (``git mv``, no
behaviour change):

  * :class:`ACTEvalVideo`
  * :class:`HPTEvalVideo`
  * :class:`PIEvalVideo`
"""

from egomimic.eval.zoo.eval_act import ACTEvalVideo
from egomimic.eval.zoo.eval_hpt import HPTEvalVideo
from egomimic.eval.zoo.eval_pi import PIEvalVideo

__all__ = ["ACTEvalVideo", "HPTEvalVideo", "PIEvalVideo"]
