"""BC-RNN: behavior cloning with a recurrent policy.

Faithful port of robomimic's ``BC_RNN`` (robomimic/algo/bc.py::BC_RNN).
Same architecture: multi-modal obs encoder -> LSTM/GRU -> per-step linear
action head, per-step MSE loss. The Gaussian-Mixture variant (BC_RNN_GMM)
is left for a follow-up.
"""

from egomimic.algo.bcrnn.algo import BCRNN
from egomimic.algo.bcrnn.outer_stage import BCRNNOuterStage

__all__ = ["BCRNN", "BCRNNOuterStage"]
