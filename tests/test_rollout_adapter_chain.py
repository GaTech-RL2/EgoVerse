"""Decoder output -> rollout adapter -> native action -> env.step().

Nothing else covers this seam. The config preflight stops at `pred_action`;
the SR gate goes from the tokenizer straight to the env without passing through
the adapter. Between them sits the `append`-layout contract, and a mis-shaped
command is not rejected by the simulator — just misinterpreted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("pymunk")

from Tsimulation.pushshapes import PushShapesEnv  # noqa: E402
from Tsimulation.pushshapes.agents import CONTROL_GAPS  # noqa: E402
from egomimic.pipeline.pushshapes import PlanarArcRolloutAdapter  # noqa: E402
from egomimic.pipeline.stages_ar import ARActionDecoder  # noqa: E402

DOMAIN = "pushshapes_sim_gripper"
HORIZON, ACTION_DIM, COND_DIM, N_WAYPOINTS = 17, 5, 67, 16
VARIANTS = ("causal_bidir", "state_action_ar", "state_idm")


def _token(variant: str) -> torch.Tensor:
    dec = ARActionDecoder(
        condition_input_dim=COND_DIM, action_horizon=HORIZON,
        action_dims={DOMAIN: ACTION_DIM}, variant=variant, d_model=64,
        n_layers=2, n_heads=4, dropout=0.0, n_waypoints=N_WAYPOINTS,
        gradient_checkpointing=False,
    ).eval()
    with torch.no_grad():
        return dec({"condition": torch.randn(1, COND_DIM),
                    "embodiment": DOMAIN})["pred_action"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_decoded_token_is_executable_by_the_simulator(variant):
    token = _token(variant)
    adapter = PlanarArcRolloutAdapter(embodiment=DOMAIN,
                                      velocity_layout="append")
    native = adapter.decode(token[0])
    assert native.shape == (N_WAYPOINTS, adapter.native_width)

    env = PushShapesEnv(object_shape="T", pusher_shape="gripper",
                        obstacle_level=0, image_size=96)
    env.reset(seed=0)
    env.agent.control_gap = CONTROL_GAPS["jittery"]
    env.agent.randomize_gap = False
    env.agent.reset_control_gap(env)
    env._skip_obs_render = True
    env.step(np.asarray(native[0], dtype=np.float64))


def test_layout_mismatch_changes_the_action_count_silently():
    """Documents the cost of getting velocity_layout wrong.

    `append`'s trailing row is a velocity summary, not a waypoint. Reading the
    same token as `concat` turns it into a 17th waypoint and the simulator
    executes it as a pose — no error, just a wrong command.
    """
    token = _token("causal_bidir")[0]
    n_append = PlanarArcRolloutAdapter(
        embodiment=DOMAIN, velocity_layout="append").decode(token).shape[0]
    n_concat = PlanarArcRolloutAdapter(
        embodiment=DOMAIN, velocity_layout="concat").decode(token).shape[0]
    assert n_append == N_WAYPOINTS
    assert n_concat == N_WAYPOINTS + 1
