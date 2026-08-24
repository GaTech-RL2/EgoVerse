"""PushShapes dataset schema for single-observation Pipeline policies."""


def get_keymap_hpt(action_horizon: int = 16, norm_mode: bool = False, **kwargs):
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
            "zarr_key": "actions",
            "horizon": int(action_horizon),
        },
    }
    if norm_mode:
        keymap.pop("front_img_1")
    return keymap


def get_rotvec_transform_list(keys: list[str] | None = None, angle_col: int = 2):
    """Encode U-socket ``theta`` targets as ``(cos(theta), sin(theta))``."""
    from egomimic.rldb.zarr.action_chunk_transforms import ThetaToRotVec

    return [ThetaToRotVec(keys=keys or ["actions"], angle_col=angle_col)]


def get_rotvec_revert_transform_list(keys: list[str] | None = None, angle_col: int = 2):
    """Decode U-socket rotation vectors before simulator consumption."""
    from egomimic.rldb.zarr.action_chunk_transforms import RotVecToTheta

    return [RotVecToTheta(keys=keys or ["actions"], angle_col=angle_col)]
