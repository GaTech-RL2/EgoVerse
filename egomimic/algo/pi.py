"""Backward-compat shim — ``egomimic.algo.pi`` moved to
``egomimic.algo.zoo.pi`` (DESIGN.md step 8, zoo home).

Kept ALIVE as a thin re-export so the legacy import path + yaml
``_target_: egomimic.algo.pi.PI`` keep working unchanged until the final flip
(DESIGN.md step 13). DO NOT add logic here — pure forwarder.
"""

from egomimic.algo.zoo.pi import *  # noqa: F401,F403
from egomimic.algo.zoo.pi import PI  # noqa: F401

__all__ = ["PI"]
