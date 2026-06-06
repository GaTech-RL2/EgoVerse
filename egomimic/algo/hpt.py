"""Backward-compat shim — ``egomimic.algo.hpt`` moved to
``egomimic.algo.zoo.hpt`` (DESIGN.md step 8, zoo home).

Kept ALIVE as a thin re-export so the legacy import path + yaml
``_target_: egomimic.algo.hpt.HPT`` keep working unchanged until the final flip
(DESIGN.md step 13). DO NOT add logic here — pure forwarder.
"""

from egomimic.algo.zoo.hpt import *  # noqa: F401,F403
from egomimic.algo.zoo.hpt import HPT, HPTModel  # noqa: F401

__all__ = ["HPT", "HPTModel"]
