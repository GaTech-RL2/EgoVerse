"""Backward-compat import shim (DESIGN.md step 5, amended).

The BC algo moved from ``egomimic/algo/bc_rnn.py`` to the flat
``egomimic/algo/bc.py`` and the classes were renamed
``BCRNN -> WindowedBC`` / ``BCRNNPolicy -> WindowedBCPolicy`` (the old names are
kept as aliases in ``bc.py``). This module re-exports the full public surface so
the legacy import path ``egomimic.algo.bc_rnn`` (and any ``_target_:
egomimic.algo.bc_rnn.BCRNN`` config) keeps resolving to the exact same objects.

Prefer importing from ``egomimic.algo.bc`` directly in new code.
"""

from egomimic.algo.bc import (  # noqa: F401
    BCRNN,
    BCRNNPolicy,
    WindowedBC,
    WindowedBCPolicy,
    _cut_windows,
    _cut_windows_strided,
    _pack_to_padded,
)

__all__ = [
    "WindowedBC",
    "WindowedBCPolicy",
    "BCRNN",
    "BCRNNPolicy",
    "_cut_windows",
    "_cut_windows_strided",
    "_pack_to_padded",
]
