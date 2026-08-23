"""Algorithm exports without eagerly importing every training stack.

Rollout only needs the base ``Algo`` contract. Importing HPT and ACT here used
to pull their full optional dependency trees into any PipelineAlgo process.
"""

from egomimic.algo.algo import Algo as Algo

__all__ = ["Algo", "ACT", "HPT"]


def __getattr__(name):
    if name == "ACT":
        from egomimic.algo.act import ACT

        return ACT
    if name == "HPT":
        from egomimic.algo.hpt import HPT

        return HPT
    raise AttributeError(name)
