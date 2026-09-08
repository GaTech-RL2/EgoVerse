"""HPT policy video-evaluator package.

Pairs with the :class:`egomimic.algo.hpt.HPT` algo. The class lives in
``eval_hpt.py``; re-exported here so the public import path is
``egomimic.eval.hpt.HPTEvalVideo``.
"""

from egomimic.eval.hpt.eval_hpt import HPTEvalVideo

__all__ = ["HPTEvalVideo"]
