"""H-Net algo package — the packed-mode family (peer of ``bc``).

Holds the packed-mode algo base + the stage-based H-Net pieces:

* :class:`PackedAlgoBase` — generic packed-mode algo; used *directly* as the
  H-Net algo (the ``hnet_pushshapes*`` configs target it) and subclassed by
  :class:`egomimic.algo.bc.WindowedBC`.
* :class:`HNetPolicy`     — the (legacy) flat H-Net policy.
* :class:`HNetLoss`       — action-MSE + ratio-loss regulariser.
* :class:`GMMLoss`        — GMM-NLL + ratio-loss (chunked GMM head).
* :class:`HNetOuterStage` — the H-Net OuterStage (encode → stage-tree → decode).

The classes live in ``algo.py``; re-exported here so the import path is
``egomimic.algo.hnet.PackedAlgoBase`` etc.
"""

from egomimic.algo.hnet.algo import (
    GMMLoss,
    HNet,
    HNetLoss,
    HNetOuterStage,
    HNetPolicy,
    PackedAlgoBase,
)

__all__ = [
    "PackedAlgoBase",
    "HNetPolicy",
    "HNetLoss",
    "GMMLoss",
    "HNetOuterStage",
    "HNet",
]
