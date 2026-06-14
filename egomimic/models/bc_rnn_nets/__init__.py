"""BC-RNN network components (robomimic recipe).

A clean, self-contained package for the BC-RNN policy, kept separate from the
H-Net code (``egomimic.models.hnet_nets``). Pieces:

* ``ObsEncoder``    -- per-frame obs (agent xy + front image) -> embedding.
* ``LSTMCore``      -- LSTM over the obs-embedding sequence (the history
                       mechanism; obs-only, NO past-action input).
* ``TransformerCore`` -- drop-in swap for ``LSTMCore``: causal self-attention
                       over the same rnn_horizon obs window (same interface;
                       obs-only, NO past-action input). Selected via the
                       ``core: transformer`` config knob.
* ``HNetCore``      -- drop-in swap for ``LSTMCore``: the user's REAL
                       hierarchical dynamic-chunking H-Net (``hnet_nets`` stage
                       tree: outer encoder/decoder around a ChunkerStage that
                       dynamically chunks over frames, with an inner
                       ComputeStage in the compressed space) over the same
                       rnn_horizon obs window (same interface; obs-only, NO
                       past-action input; CAUSAL). Selected via ``core: hnet``.
* ``GMMActionHead`` -- diagonal Gaussian-mixture head decoding ONE action per
                       step from the core's per-step feature (NLL train,
                       mode-commit decode).

Vision backbones (``SimpleConv`` / ``ResNetEncoder``) are reused from
``egomimic.models.hnet_nets.image_encoders`` via Hydra ``_target_`` in the
config; nothing in the H-Net *algo* path is imported here.
"""

from egomimic.models.bc_rnn_nets.gmm_head import GMMActionHead
from egomimic.models.bc_rnn_nets.hnet_core import HNetCore
from egomimic.models.bc_rnn_nets.lstm_core import LSTMCore
from egomimic.models.bc_rnn_nets.obs_encoder import ObsEncoder
from egomimic.models.bc_rnn_nets.query_decoder import QueryActionDecoder
from egomimic.models.bc_rnn_nets.transformer_core import TransformerCore

__all__ = [
    "ObsEncoder",
    "LSTMCore",
    "TransformerCore",
    "HNetCore",
    "GMMActionHead",
    "QueryActionDecoder",
]
