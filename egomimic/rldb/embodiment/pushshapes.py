"""PushShapes dataset schema for single-observation Pipeline policies."""


def get_keymap_hpt(
    action_horizon: int = 16,
    norm_mode: bool = False,
    action_zarr_key: str = "actions",
    **kwargs,
):
    """Map one current observation to a future action chunk.

    Observation keys intentionally have no horizon. Exposing future
    observations would leak the target during training and mismatch rollout.
    """
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
            "horizon": int(action_horizon),
        },
    }
    if norm_mode:
        keymap.pop("front_img_1")
    return keymap


def get_chain_gripper_point_validation_transform_list(
    keys: list[str] | None = None,
):
    """Validate direct ChainGripper point targets before normalization."""
    from egomimic.rldb.zarr.action_chunk_transforms import RequireLastDim

    return [RequireLastDim(keys=keys or ["actions"], width=6)]


def get_chain_gripper_point_transform_list(
    keys: list[str] | None = None,
    world_size: float = 512.0,
):
    """Derive ordered point targets from native ChainGripper controls."""
    from egomimic.rldb.zarr.action_chunk_transforms import (
        ChainGripperNative4ToPoints6,
    )

    return [
        ChainGripperNative4ToPoints6(
            keys=keys or ["actions"],
            world_size=world_size,
        )
    ]


def get_chain_gripper_point_revert_transform_list(
    keys: list[str] | None = None,
    world_size: float = 512.0,
    grid_size: int = 33,
    refinements: int = 6,
    context_state_key: str = "state_agent_obj",
    previous_control_key: str = "previous_control",
):
    """Project model point predictions back to native ChainGripper controls."""
    from egomimic.rldb.zarr.action_chunk_transforms import (
        ChainGripperPoints6ToNative4,
    )

    return [
        ChainGripperPoints6ToNative4(
            keys=keys or ["actions"],
            world_size=world_size,
            grid_size=grid_size,
            refinements=refinements,
            context_state_key=context_state_key,
            previous_control_key=previous_control_key,
        )
    ]


def get_rotvec_transform_list(keys: list[str] | None = None, angle_col: int = 2):
    """Encode U-socket ``theta`` targets as ``(cos(theta), sin(theta))``."""
    from egomimic.rldb.zarr.action_chunk_transforms import ThetaToRotVec

    return [ThetaToRotVec(keys=keys or ["actions"], angle_col=angle_col)]


def get_rotvec_revert_transform_list(keys: list[str] | None = None, angle_col: int = 2):
    """Decode U-socket rotation vectors before simulator consumption."""
    from egomimic.rldb.zarr.action_chunk_transforms import RotVecToTheta

    return [RotVecToTheta(keys=keys or ["actions"], angle_col=angle_col)]


def get_arc_length_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 200.0,
    resampled_vector_length: int = 25,
    dt: float = 1.0 / 30.0,
    rotation_radius: float = 40.0,
):
    """Create the planar SE(2) arc-length transform for U-socket actions."""
    from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeUSocketArcLength

    action_keys = keys or ["actions"]
    if len(action_keys) != 1:
        raise ValueError("U-socket arc-length tokenization expects exactly one key")
    return [
        TokenizeUSocketArcLength(
            action_key=action_keys[0],
            output_action_key=action_keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
        )
    ]


def get_chain_gripper_point_arc_length_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 200.0,
    resampled_vector_length: int = 25,
    dt: float = 1.0 / 30.0,
):
    """Create the three-point planar arc transform for ChainGripper actions."""
    from egomimic.rldb.zarr.arc_length_tokenizer import (
        TokenizeChainGripperPointArcLength,
    )

    action_keys = keys or ["actions"]
    if len(action_keys) != 1:
        raise ValueError("ChainGripper point arc tokenization expects exactly one key")
    return [
        TokenizeChainGripperPointArcLength(
            action_key=action_keys[0],
            output_action_key=action_keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
        )
    ]


def get_chain_gripper_native_point_arc_length_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 200.0,
    resampled_vector_length: int = 25,
    dt: float = 1.0 / 30.0,
    world_size: float = 512.0,
):
    """Compose native4-to-points FK with point-space arc tokenization."""
    action_keys = keys or ["actions"]
    return get_chain_gripper_point_transform_list(
        keys=action_keys,
        world_size=world_size,
    ) + get_chain_gripper_point_arc_length_transform_list(
        keys=action_keys,
        min_distance_unit=min_distance_unit,
        resampled_vector_length=resampled_vector_length,
        dt=dt,
    )


def get_planar_dense_transform_list(keys: list[str] | None = None):
    """h16-style dense baseline, widened to the shared 5-channel layout.

    Pairs with the arc configs: identical action representation, the only
    difference being time-indexed dense chunks vs arc-length tokens. Without
    the shared layout the comparison would confound tokenization with a
    change of action space.
    """
    from egomimic.rldb.zarr.arc_length_tokenizer import PadPlanarAction

    return [PadPlanarAction(keys=keys or ["actions"])]


def get_planar_arc_length_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 50.0,
    resampled_vector_length: int = 16,
    dt: float = 1.0 / 30.0,
    rotation_radius: float = 0.0,
    hybrid_rotation_unit: float | None = None,
    velocity_mode: str = "mean_scalar",
    velocity_layout: str = "append",
    residual_xy: bool = False,
):
    """Embodiment-agnostic planar arc tokenization.

    Every knob the tokenizer exposes has to appear here or hydra cannot reach
    it: a config key with no matching parameter raises TypeError only once the
    job is on the node, after the data pull. That cost 8 of 9 sweep runs.

      rotation_radius       0 measures arc by translation alone and lets theta
                            ride the path; >0 adds lambda_for_radius(r)*mu to
                            the step, so rotating costs what translating r
                            units costs.
      hybrid_rotation_unit  gives rotation its own budget; the token then spans
                            min(D/v_trans, D_rot/v_rot).
      velocity_mode         mean_scalar | per_step_scalar.
      velocity_layout       append -> (M+1, 5); concat -> (M, 5+V).
      residual_xy           express xy waypoints relative to the pusher's
                            current position. PushShapes actions are ABSOLUTE
                            targets equal to the pusher's own position plus a
                            <=3px delta, and that position is an input channel,
                            so a persistence baseline already scores FVE
                            0.85-0.95 and MSE on the absolute target learns
                            almost nothing about the delta that drives the task.
                            Changes the norm-stat scale too (~10px instead of
                            ~500px), which is the half that actually matters.
                            A model trained with this ON must be rolled out with
                            PipelineAlgo(residual_xy=True) or it will command
                            deltas as if they were absolute world coordinates.
    """
    from egomimic.rldb.zarr.arc_length_tokenizer import (
        ResidualizeArcTokenXY, TokenizePlanarArcLength)

    action_keys = keys or ["actions"]
    if len(action_keys) != 1:
        raise ValueError("planar arc tokenization expects exactly one action key")
    transforms = [
        TokenizePlanarArcLength(
            action_key=action_keys[0],
            output_action_key=action_keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            rotation_radius=rotation_radius,
            hybrid_rotation_unit=hybrid_rotation_unit,
            velocity_mode=velocity_mode,
            velocity_layout=velocity_layout,
        )
    ]
    if residual_xy:
        transforms.append(
            ResidualizeArcTokenXY(
                action_key=action_keys[0],
                state_key="state_agent_obj",
                velocity_layout=velocity_layout,
            )
        )
    return transforms
