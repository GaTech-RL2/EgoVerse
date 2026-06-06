"""Backward-compat shim — ``egomimic.algo.act`` moved to
``egomimic.algo.zoo.act`` (DESIGN.md step 8, zoo home).

Kept ALIVE as a thin re-export so the legacy import path + yaml
``_target_: egomimic.algo.act.ACT`` keep working unchanged until the final flip
(DESIGN.md step 13). DO NOT add logic here — pure forwarder.
"""

from egomimic.algo.zoo.act import *  # noqa: F401,F403
from egomimic.algo.zoo.act import ACT, ACTModel  # noqa: F401

__all__ = ["ACT", "ACTModel"]
