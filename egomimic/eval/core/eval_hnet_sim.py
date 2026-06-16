"""HNet sim-eval alias. Ported from EgoVerse2 eval/eval_hnet_sim.py — re-exports
the canonical sim-rollout evaluator under the HNetSimEval name used by some
EV2 evaluator configs."""
from egomimic.eval.core.eval_sim import SimRolloutEval as HNetSimEval

__all__ = ["HNetSimEval"]
