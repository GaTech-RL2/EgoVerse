"""PushShapes: T/U/Z pushing env with multiple pushers and obstacle levels."""

from gymnasium.envs.registration import register

from Tsimulation.pushshapes.env import PushShapesEnv

register(
    id="PushShapes-v0",
    entry_point="Tsimulation.pushshapes.env:PushShapesEnv",
    max_episode_steps=300,
)

__all__ = ["PushShapesEnv"]
