"""PushShapes sim-eval glue: env-output <-> canonical zarr-key conversion.

These helpers translate the ``Tsimulation.pushshapes.PushShapesEnv`` native
obs into the canonical zarr-key dict the dataset emits (so the algo's keymap +
transforms apply unchanged), and split a flat state vector back into the env's
``set_state`` args for replay-mode init. They are pushshapes-embodiment-specific
glue and therefore live with the embodiment, not in the algo-agnostic
``eval/core/eval_sim.py`` evaluator.

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


def _env_to_zarr_pushshapes_usocket(obs_env: dict, device: torch.device) -> dict:
    """u_socket variant: state_agent_obj (1, 6) = concat(pusher_xy, pusher_angle,
    obj_xyangle) -- matches the u_socket training zarr layout, where the emb-19
    obs spec slices [0:3] = pusher (x, y, theta). The generic 5-dim converter
    would silently feed (x, y, obj_x) into that slice."""
    state_6 = np.concatenate(
        [obs_env["agent_pos"], obs_env["agent_angle"], obs_env["object_pose"]],
        axis=0,
    ).astype(np.float32)
    image_chw = np.transpose(obs_env["image"], (2, 0, 1)).astype(np.float32) / 255.0
    return {
        "state_agent_obj": torch.from_numpy(state_6).unsqueeze(0).to(device),
        "front_img_1": torch.from_numpy(image_chw).unsqueeze(0).to(device),
    }


_ENV_TO_ZARR = {
    "pushshapes_sim": _env_to_zarr_pushshapes,
    "pushshapes_sim:u_socket": _env_to_zarr_pushshapes_usocket,
}


def _state_to_init(state: np.ndarray) -> tuple:
    """Split (5,) state vector into PushShapesEnv set_state args.
    state = concat(pusher_xy, object_xyangle).
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 5:
        raise ValueError(f"expected state of len >= 5, got {state.shape}")
    return (
        (float(state[0]), float(state[1])),
        (float(state[2]), float(state[3]), float(state[4])),
    )
