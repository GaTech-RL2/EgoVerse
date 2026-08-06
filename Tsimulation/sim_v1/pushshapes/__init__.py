"""PushShapes: T/U/Z pushing env with multiple pushers and obstacle levels."""

from gymnasium.envs.registration import register

from Tsimulation.sim_v1.pushshapes.env import PushShapesEnv

register(
    id="PushShapes-v1",
    entry_point="Tsimulation.sim_v1.pushshapes.env:PushShapesEnv",
)

__all__ = ["PushShapesEnv"]
