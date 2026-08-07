"""ACT policy video-evaluator package.

Pairs with the :class:`egomimic.algo.act.ACT` algo. The class lives in
``eval_act.py``; re-exported here so the public import path is
``egomimic.eval.act.ACTEvalVideo``.
"""

from egomimic.eval.act.eval_act import ACTEvalVideo

__all__ = ["ACTEvalVideo"]
