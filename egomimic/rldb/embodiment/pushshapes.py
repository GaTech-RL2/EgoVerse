"""Dataset schema and transforms for the two Planar PushShapes embodiments."""

from __future__ import annotations

from egomimic.rldb.zarr.action_chunk_transforms import (
    PlanarAgentStateToRotVec4,
    ThetaToRotVec,
)
from egomimic.rldb.zarr.planar_arc import PadPlanarAction, TokenizePlanarArcLength


def get_planar_keymap(
    action_horizon: int = 16,
    observation_horizon: int = 1,
    action_target_offset: int = 0,
    norm_mode: bool = False,
    action_zarr_key: str = "actions",
    **_kwargs,
):
    """Map observations and a future native-action chunk from one episode."""
    observation_horizon = int(observation_horizon)
    action_target_offset = int(action_target_offset)
    if observation_horizon <= 0 or action_target_offset < 0:
        raise ValueError("observation_horizon must be positive and offset non-negative")
    keymap = {
        "front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "observations.images.front_img_1",
        },
        "state_agent_obj": {
            "key_type": "proprio_keys",
            "zarr_key": "observations.state",
        },
        "actions": {
            "key_type": "action_keys",
            "zarr_key": str(action_zarr_key),
            "horizon": int(action_horizon) + action_target_offset,
        },
    }
    if observation_horizon > 1:
        keymap["front_img_1"]["horizon"] = observation_horizon
        keymap["state_agent_obj"]["horizon"] = observation_horizon
    if norm_mode:
        keymap.pop("front_img_1")
    return keymap


def get_planar_keymap_per_source_proprio(
    action_horizon: int = 16,
    norm_mode: bool = False,
    action_zarr_key: str = "actions",
    **kwargs,
):
    """Keep native simulator state as metadata and add model-only proprio."""
    keymap = get_planar_keymap(
        action_horizon=action_horizon,
        norm_mode=norm_mode,
        action_zarr_key=action_zarr_key,
        **kwargs,
    )
    keymap["state_agent_obj"]["key_type"] = "metadata_keys"
    keymap["state_agent_model"] = {
        "key_type": "proprio_keys",
        "zarr_key": "observations.state",
    }
    return keymap


class SliceActionTarget:
    """Align the target to the final observation in a multi-frame window."""

    def __init__(self, keys: list[str], start: int, horizon: int):
        self.keys = list(keys)
        self.start = int(start)
        self.horizon = int(horizon)
        if self.start < 0 or self.horizon <= 0:
            raise ValueError("start must be non-negative and horizon positive")

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            value = batch[key][self.start : self.start + self.horizon]
            if len(value) != self.horizon:
                raise ValueError(
                    f"{key!r} has {len(value)} aligned steps, expected {self.horizon}"
                )
            batch[key] = value
        return batch


def get_planar_dense_transform_list(keys: list[str] | None = None, **_kwargs):
    """Convert either native Planar action layout into common five-space."""
    return [PadPlanarAction(keys=keys or ["actions"])]


def get_planar_paper_transform_list(
    keys: list[str] | None = None,
    action_horizon: int = 16,
    action_target_offset: int = 1,
    **_kwargs,
):
    """Align Paper-DP targets then convert them into common five-space."""
    keys = keys or ["actions"]
    return [
        SliceActionTarget(keys, start=action_target_offset, horizon=action_horizon),
        PadPlanarAction(keys),
    ]


def get_planar_arc_length_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 200.0,
    resampled_vector_length: int = 100,
    dt: float = 1.0 / 30.0,
    rotation_radius: float = 0.0,
    hybrid_rotation_unit: float | None = None,
    **_kwargs,
):
    """Create the active Planar SE(2) arc transform."""
    keys = keys or ["actions"]
    if len(keys) != 1:
        raise ValueError("Planar arc tokenization requires exactly one action key")
    return [
        TokenizePlanarArcLength(
            action_key=keys[0],
            output_action_key=keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            rotation_radius=rotation_radius,
            hybrid_rotation_unit=hybrid_rotation_unit,
        )
    ]


def get_usocket_rotvec_action_state_transform_list(
    action_key: str = "actions",
    state_key: str = "state_agent_model",
):
    """Encode action theta and the observed U-Socket agent pose."""
    return [
        ThetaToRotVec(keys=[action_key], angle_col=2),
        PlanarAgentStateToRotVec4(keys=[state_key], angle_col=2),
    ]
