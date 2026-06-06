"""Temporal CORE modules (DESIGN.md §2 ``models/cores/``).

Role home for the interchangeable history mechanisms that sit between the stem
and the head — the drop-in-swappable trunk cores that share the
``forward``/``init_hidden``/``step`` contract so the BC policy is core-agnostic:

  * :class:`LSTMCore`        — robomimic BC-RNN LSTM recurrence.
  * :class:`TransformerCore` — causal self-attention over the obs window.
  * :class:`HNetCore`        — the real dynamic-chunking H-Net (imports the
                               canonical ``egomimic.models.hnet`` superset tree,
                               flipped in DESIGN.md step 4).

Relocated here from ``models/bc_rnn_nets`` in DESIGN.md step 6 via ``git mv``
(no behaviour change); the old ``egomimic.models.bc_rnn_nets`` import path stays
alive through the ``bc_rnn_nets`` facade shim until DESIGN.md step 13.
"""

from egomimic.models.cores.hnet_core import HNetCore
from egomimic.models.cores.lstm_core import LSTMCore
from egomimic.models.cores.transformer_core import TransformerCore

__all__ = ["LSTMCore", "TransformerCore", "HNetCore"]
