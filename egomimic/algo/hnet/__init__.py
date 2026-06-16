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

# EV2-ported flat-fused + chunk-token H-Net families (new classes; do not
# clash with gmm's PackedAlgoBase/HNetPolicy above). ``fused`` carries EV2's
# own HNet-algo machinery internally; only the public new classes are exported.
from egomimic.algo.hnet.fused import FlatFusedPolicy, HNetFused
from egomimic.algo.hnet.chunk import (
    ChunkTokenPolicy,
    FlowHead,
    HNetChunkToken,
)

__all__ = [
    "PackedAlgoBase",
    "HNetPolicy",
    "HNetLoss",
    "GMMLoss",
    "HNetOuterStage",
    "HNet",
    "FlatFusedPolicy",
    "HNetFused",
    "ChunkTokenPolicy",
    "HNetChunkToken",
    "FlowHead",
]
