"""Backward-compat import shim (PHASE 1, structural fixes).

The H-Net family algo base moved from ``egomimic/algo/hnet.py`` to
``egomimic/algo/packed_base.py`` (a role-clarifying rename: this is the
per-embodiment-norm + packed-path base that the BC / hourglass algos subclass,
NOT the ``models/hnet/`` stage tree). **Class names are unchanged** — ``HNet``
is still ``HNet``. This module re-exports the full public surface so the legacy
import path ``egomimic.algo.hnet`` and every ``_target_: egomimic.algo.hnet.*``
config keep resolving to the exact same objects.

Prefer importing from ``egomimic.algo.packed_base`` directly in new code. This
shim is removed at the DESIGN.md step-13 final flip.
"""

from egomimic.algo.packed_base import (  # noqa: F401
    FlatFusedPolicy,
    HNet,
    HNetFused,
    HNetPolicy,
)

__all__ = ["HNet", "HNetFused", "HNetPolicy", "FlatFusedPolicy"]
