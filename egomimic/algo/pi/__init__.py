"""PI (pi-0 / pi-0.5) ALGO package.

A first-class, actively-developed comparison policy — a peer of ``bc`` — that
shares the :class:`egomimic.algo.algo.Algo` spine but is NOT part of the
hourglass line. The class lives in ``pi.py``; this package re-exports it so the
public import path is ``egomimic.algo.pi.PI``.

Importing this package pulls in ``openpi`` (an optional dep), so only import it
when the PI algo is actually needed — the top-level ``egomimic.algo`` package
does NOT import it eagerly.
"""

from egomimic.algo.pi.pi import PI

__all__ = ["PI"]
