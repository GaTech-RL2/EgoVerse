"""BC-RNN network components (robomimic recipe).

A clean, self-contained package for the BC-RNN policy. Pieces:

* ``ObsEncoder``    -- per-frame obs (agent xy + front image) -> embedding.
* ``LSTMCore``      -- LSTM over the obs-embedding sequence (the history
                       mechanism; obs-only, NO past-action input).
* ``TransformerCore`` -- drop-in swap for ``LSTMCore``: causal self-attention
                       over the same rnn_horizon obs window (same interface;
                       obs-only, NO past-action input). Selected via the
                       ``core: transformer`` config knob.
* ``HNetCore``      -- drop-in swap for ``LSTMCore``: the user's REAL
                       hierarchical dynamic-chunking H-Net (an outer
                       encoder/decoder around a ChunkerStage that dynamically
                       chunks over frames, with an inner ComputeStage in the
                       compressed space) over the same rnn_horizon obs window
                       (same interface; obs-only, NO past-action input; CAUSAL).
                       Selected via ``core: hnet``. NOTE (EgoVerse-pact-2,
                       DESIGN.md steps 3-4): HNetCore now imports the CANONICAL
                       H-Net stage tree at ``egomimic.models.hnet`` -- the pact
                       SUPERSET (cross-attn + ``residual_scale`` +
                       ``causal_conv1d`` + ``adaln_per_token``; verified the
                       cross-attn / AdaLN / window flags all default OFF, so the
                       obs-only core never touches the diverged paths and its
                       fixed-seed forward is bit-identical to the old vendored
                       copy). The former local
                       ``bc_rnn_nets._hnet_vendored`` dup (an inferior subset)
                       was deleted in the collapse; the legacy
                       ``egomimic.models.hnet_nets`` import path stays alive via
                       a thin facade shim until DESIGN.md step 13.
* ``QueryActionDecoder`` -- ACT/HPT-style action-query chunk readout: chunk_len
                       learnable queries refined by self-attn(queries) + causal
                       cross-attn over the core's per-step features, then a
                       SHARED projection to per-step GMM params. Selected via
                       ``chunk_head: queries`` (transformer core only).
* ``GMMActionHead`` -- diagonal Gaussian-mixture head decoding ONE action per
                       step (or chunk_len actions when chunk_len>1) from the
                       core's per-step feature (NLL train, mode-commit decode).
* ``VisualCore`` / ``SpatialSoftmax`` -- robomimic image backbone (ResNet18 ->
                       SpatialSoftmax -> Linear) with optional crop aug. Lives
                       HERE (not in ``hnet_nets.image_encoders``) because the
                       EgoVerse-pact tree's image_encoders.py diverged and lacks
                       VisualCore; keeping it in the BC-RNN package makes this
                       port self-contained. The model configs point
                       ``front_img_1._target_`` at
                       ``egomimic.models.bc_rnn_nets.visual_core.VisualCore``.
"""

from egomimic.models.bc_rnn_nets.gmm_head import GMMActionHead
from egomimic.models.bc_rnn_nets.hnet_core import HNetCore
from egomimic.models.bc_rnn_nets.lstm_core import LSTMCore
from egomimic.models.bc_rnn_nets.obs_encoder import ObsEncoder
from egomimic.models.bc_rnn_nets.query_decoder import QueryActionDecoder
from egomimic.models.bc_rnn_nets.transformer_core import TransformerCore
from egomimic.models.bc_rnn_nets.visual_core import SpatialSoftmax, VisualCore

__all__ = [
    "ObsEncoder",
    "LSTMCore",
    "TransformerCore",
    "HNetCore",
    "GMMActionHead",
    "QueryActionDecoder",
    "VisualCore",
    "SpatialSoftmax",
]
