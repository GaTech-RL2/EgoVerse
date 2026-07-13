"""PI policy video-evaluator package.

Pairs with the :class:`egomimic.algo.pi.PI` algo. The class lives in
``eval_pi.py``; re-exported here so the public import path is
``egomimic.eval.pi.PIEvalVideo``.
"""

from egomimic.eval.pi.eval_pi import PIEvalVideo

__all__ = ["PIEvalVideo"]
