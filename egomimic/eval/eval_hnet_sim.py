"""
Back-compat shim: the H-Net-specific sim evaluator has been generalized
into :class:`egomimic.eval.eval_sim.SimRolloutEval`. This module exists
only so existing yaml configs / scripts that import ``HNetSimEval`` keep
working.

New code should use ``SimRolloutEval`` from ``eval_sim`` directly. Each
algo implements ``sim_init_state`` + ``sim_predict_step`` and the eval
class is algo-agnostic.
"""

from egomimic.eval.eval_sim import SimRolloutEval as HNetSimEval

__all__ = ["HNetSimEval"]
