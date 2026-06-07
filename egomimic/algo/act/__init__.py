"""ACT (action-chunking transformer) ALGO package.

A first-class, actively-developed comparison policy — a peer of ``bc`` — that
shares the :class:`egomimic.algo.algo.Algo` spine but is NOT part of the
hourglass line. The class lives in ``act.py``; this package re-exports it so the
public import path is ``egomimic.algo.act.ACT``.
"""

from egomimic.algo.act.act import ACT, ACTModel

__all__ = ["ACT", "ACTModel"]
