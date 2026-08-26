"""PushShapes sim-eval glue: env-output <-> canonical zarr-key conversion.

These helpers translate the ``Tsimulation.pushshapes.PushShapesEnv`` native
obs into the canonical zarr-key dict the dataset emits (so the algo's keymap +
transforms apply unchanged), and split a flat state vector back into the env's
``set_state`` args for replay-mode init. Oriented agents keep their angle in
the state vector; free XY pushers do not. These helpers are
pushshapes-embodiment-specific glue and therefore live with the embodiment,
not in the algo-agnostic ``eval/core/eval_sim.py`` evaluator.

Extracted (byte-identical) from ``egomimic/eval/core/eval_sim.py`` during the
eval+pl_utils hierarchy pass; ``eval_sim`` now imports ``_env_to_zarr_pushshapes``,
``_state_to_init`` and ``_ENV_TO_ZARR`` from here. The legacy import paths
``egomimic.eval.eval_sim._env_to_zarr_pushshapes`` / ``._state_to_init`` and
``egomimic.eval.core.eval_sim._state_to_init`` keep resolving because those
names are re-exported back into ``eval_sim``'s namespace.
"""

from __future__ import annotations

import numpy as np
import torch


def _env_to_zarr_pushshapes(obs_env: dict, device: torch.device) -> dict:
    """PushShapesEnv obs -> canonical zarr-key dict (B=1).

    Keys match pushshapes.get_keymap:
      state_agent_obj: (1, 5) = concat(pusher_xy, obj_xyangle)
      front_img_1: (1, 3, H, W) float [0,1]
    """
    state_5 = np.concatenate(
        [obs_env["agent_pos"], obs_env["object_pose"]], axis=0
    ).astype(np.float32)
    image_chw = np.transpose(obs_env["image"], (2, 0, 1)).astype(np.float32) / 255.0
    return {
        "state_agent_obj": torch.from_numpy(state_5).unsqueeze(0).to(device),
        "front_img_1": torch.from_numpy(image_chw).unsqueeze(0).to(device),
    }


def _env_to_zarr_pushshapes_oriented(obs_env: dict, device: torch.device) -> dict:
    """PushShapesEnv obs for a controlled-angle agent -> dataset keys.

    U-Socket and ChainGripper datasets store
    ``[agent_x, agent_y, agent_angle, object_x, object_y, object_angle]``.
    Keeping this as a separate converter prevents a silent five-vs-six state
    mismatch for the historical circle embodiments.
    """
    state_6 = np.concatenate(
        [
            obs_env["agent_pos"],
            obs_env["agent_angle"],
            obs_env["object_pose"],
        ],
        axis=0,
    ).astype(np.float32)
    image_chw = np.transpose(obs_env["image"], (2, 0, 1)).astype(np.float32) / 255.0
    return {
        "state_agent_obj": torch.from_numpy(state_6).unsqueeze(0).to(device),
        "front_img_1": torch.from_numpy(image_chw).unsqueeze(0).to(device),
    }


_ENV_TO_ZARR = {
    "pushshapes_sim": _env_to_zarr_pushshapes,
    "pushshapes_sim_small_circle": _env_to_zarr_pushshapes,
    "pushshapes_sim_u_socket": _env_to_zarr_pushshapes_oriented,
    "pushshapes_sim_chain_gripper": _env_to_zarr_pushshapes_oriented,
}


def _state_to_init(state: np.ndarray) -> tuple:
    """Backward-compatible ``(agent_pos, object_pose)`` state split.

    Five-dimensional states are ``XY + object pose``. Six-dimensional states
    are ``XY + controlled angle + object pose``; the controlled angle is
    intentionally omitted from this legacy return value. New replay code
    should call :func:`_state_to_env_init` so it cannot discard orientation.
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 5:
        raise ValueError(f"expected state of len >= 5, got {state.shape}")
    object_offset = 3 if state.shape[0] >= 6 else 2
    return (
        (float(state[0]), float(state[1])),
        tuple(float(value) for value in state[object_offset : object_offset + 3]),
    )


def _state_to_env_init(state: np.ndarray, embodiment_name: str) -> dict:
    """Split a stored state into the exact arguments accepted by ``set_state``."""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    oriented = embodiment_name in {
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    }
    expected = 6 if oriented else 5
    if state.shape[0] != expected:
        raise ValueError(
            f"{embodiment_name} expects state width {expected}, got {state.shape}"
        )
    object_offset = 3 if oriented else 2
    result = {
        "agent_pos": (float(state[0]), float(state[1])),
        "object_pose": tuple(
            float(value) for value in state[object_offset : object_offset + 3]
        ),
    }
    if oriented:
        result["agent_angle"] = float(state[2])
    return result
