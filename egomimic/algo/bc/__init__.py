"""BC (behaviour-cloning) ALGO package.

A first-class comparison policy — peer of ``hpt`` / ``act`` / ``pi`` — built on
the packed-mode :class:`egomimic.algo.hnet.PackedAlgoBase` spine as an
EXACT-REPLICA of robomimic BC_RNN_GMM. The classes live in ``algo.py``; this
package re-exports them so the public import path stays
``egomimic.algo.bc.WindowedBC``.
"""

from egomimic.algo.bc.algo import WindowedBC, WindowedBCPolicy

__all__ = ["WindowedBC", "WindowedBCPolicy"]
