from typing import Any, Dict, Tuple

import torch

PI05_CARTESIAN_ACTION_ENCODING_RAW_ROT_6D = "cartesian_ypr_raw_rot6d"
PI05_CARTESIAN_ACTION_ENCODING_LEGACY = "legacy_normalized_ypr_rot6d"

# Bimanual robot Cartesian layout: [x, y, z, yaw, pitch, roll, gripper] x 2.
ROBOT_BIMANUAL_CARTESIAN_ROT_DIMS = (3, 4, 5, 10, 11, 12)
ROBOT_BIMANUAL_CARTESIAN_NON_ROT_DIMS = (0, 1, 2, 6, 7, 8, 9, 13)


# ---------- registry that stores *objects* ----------
class ConverterRegistry:
    def __init__(self):
        self._converters: Dict[Tuple[int | str, str], "BaseActionConverter"] = {}
        self._ANY = "*"

    def register(
        self, embodiment_id: int | str, ac_key: str, obj: "BaseActionConverter"
    ):
        self._converters[(embodiment_id, ac_key)] = obj

    def get(self, embodiment_id: int, ac_key: str) -> "BaseActionConverter":
        return (
            self._converters.get((embodiment_id, ac_key))
            or self._converters.get((embodiment_id, self._ANY))
            or self._converters.get((self._ANY, ac_key))
            or self._converters.get((self._ANY, self._ANY))
        )


# ---------- shared helpers (same conventions you used) ----------
def _ensure_bsd(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim != 3:
        raise ValueError(f"Expected (B,S,D), got {tuple(x.shape)}")
    return x


def _pad32(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_bsd(x)
    B, S, D = x.shape
    if D == 32:
        return x
    if D < 32:
        pad = torch.zeros(B, S, 32 - D, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)
    return x[..., :32]


def _stat_tensor(stats: dict[str, Any], key: str, ref: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(stats[key], device=ref.device, dtype=torch.float32)
    return value.to(dtype=ref.dtype if ref.is_floating_point() else torch.float32)


def _apply_norm_one(
    tensor: torch.Tensor,
    stats: dict[str, Any],
    norm_mode: str,
) -> torch.Tensor:
    if norm_mode == "zscore":
        mean = _stat_tensor(stats, "mean", tensor)
        std = _stat_tensor(stats, "std", tensor)
        return (tensor - mean) / (std + 1e-6)
    if norm_mode == "minmax":
        mn = _stat_tensor(stats, "min", tensor)
        mx = _stat_tensor(stats, "max", tensor)
        return 2.0 * ((tensor - mn) / (mx - mn + 1e-6)) - 1.0
    if norm_mode == "quantile":
        q1 = _stat_tensor(stats, "quantile_1", tensor)
        q99 = _stat_tensor(stats, "quantile_99", tensor)
        return 2.0 * ((tensor - q1) / (q99 - q1 + 1e-6)) - 1.0
    raise ValueError(f"Invalid normalization mode: {norm_mode}")


def _apply_unnorm_one(
    tensor: torch.Tensor,
    stats: dict[str, Any],
    norm_mode: str,
) -> torch.Tensor:
    if norm_mode == "zscore":
        mean = _stat_tensor(stats, "mean", tensor)
        std = _stat_tensor(stats, "std", tensor)
        return tensor * (std + 1e-6) + mean
    if norm_mode == "minmax":
        mn = _stat_tensor(stats, "min", tensor)
        mx = _stat_tensor(stats, "max", tensor)
        return (tensor + 1) * 0.5 * (mx - mn + 1e-6) + mn
    if norm_mode == "quantile":
        q1 = _stat_tensor(stats, "quantile_1", tensor)
        q99 = _stat_tensor(stats, "quantile_99", tensor)
        return (tensor + 1) * 0.5 * (q99 - q1 + 1e-6) + q1
    raise ValueError(f"Invalid normalization mode: {norm_mode}")


def _normalize_robot_bimanual_non_rot(
    raw_actions: torch.Tensor,
    stats: dict[str, Any],
    norm_mode: str,
) -> torch.Tensor:
    normalized = raw_actions.clone()
    all_dims = _apply_norm_one(raw_actions, stats, norm_mode)
    normalized[..., ROBOT_BIMANUAL_CARTESIAN_NON_ROT_DIMS] = all_dims[
        ..., ROBOT_BIMANUAL_CARTESIAN_NON_ROT_DIMS
    ]
    return normalized


def _unnormalize_robot_bimanual_non_rot(
    model_actions: torch.Tensor,
    stats: dict[str, Any],
    norm_mode: str,
) -> torch.Tensor:
    raw_actions = model_actions.clone()
    all_dims = _apply_unnorm_one(model_actions, stats, norm_mode)
    raw_actions[..., ROBOT_BIMANUAL_CARTESIAN_NON_ROT_DIMS] = all_dims[
        ..., ROBOT_BIMANUAL_CARTESIAN_NON_ROT_DIMS
    ]
    return raw_actions


def _ypr_to_matrix(ypr: torch.Tensor, degrees: bool = False) -> torch.Tensor:
    if degrees:
        ypr = ypr * (torch.pi / 180.0)
    yaw, pitch, roll = ypr.unbind(-1)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    Rz = torch.stack(
        [
            torch.stack([cy, -sy, torch.zeros_like(cy)], dim=-1),
            torch.stack([sy, cy, torch.zeros_like(cy)], dim=-1),
            torch.stack(
                [torch.zeros_like(cy), torch.zeros_like(cy), torch.ones_like(cy)],
                dim=-1,
            ),
        ],
        dim=-2,
    )
    Ry = torch.stack(
        [
            torch.stack([cp, torch.zeros_like(cp), sp], dim=-1),
            torch.stack(
                [torch.zeros_like(cp), torch.ones_like(cp), torch.zeros_like(cp)],
                dim=-1,
            ),
            torch.stack([-sp, torch.zeros_like(cp), cp], dim=-1),
        ],
        dim=-2,
    )
    Rx = torch.stack(
        [
            torch.stack(
                [torch.ones_like(cr), torch.zeros_like(cr), torch.zeros_like(cr)],
                dim=-1,
            ),
            torch.stack([torch.zeros_like(cr), cr, -sr], dim=-1),
            torch.stack([torch.zeros_like(cr), sr, cr], dim=-1),
        ],
        dim=-2,
    )
    return Rz @ Ry @ Rx  # (B,S,3,3)


def _matrix_to_ypr(R: torch.Tensor, degrees: bool = False) -> torch.Tensor:
    """
    Inverse of R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    Returns (B,S,3) [yaw, pitch, roll], radians by default.
    """
    # Clamp for numerical safety
    sy = -R[..., 2, 0]  # -sin(pitch)
    sy = sy.clamp(-1.0, 1.0)
    pitch = torch.asin(sy)

    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])

    ypr = torch.stack([yaw, pitch, roll], dim=-1)
    if degrees:
        ypr = ypr * (180.0 / torch.pi)
    return ypr


def _reconstruct_R_from_cols(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """
    Given first two columns (B,S,3), produce a proper rotation matrix:
      - normalize c1, orthogonalize c2 wrt c1, normalize c2
      - c3 = c1 x c2
    Returns R (B,S,3,3)
    """
    eps = 1e-8
    c1n = c1 / (c1.norm(dim=-1, keepdim=True).clamp_min(eps))
    # Gram-Schmidt for c2
    proj = (c2 * c1n).sum(dim=-1, keepdim=True) * c1n
    c2o = c2 - proj
    c2n = c2o / (c2o.norm(dim=-1, keepdim=True).clamp_min(eps))
    c3n = torch.cross(c1n, c2n, dim=-1)
    R = torch.stack([c1n, c2n, c3n], dim=-1)  # (B,S,3,3) as columns
    return R


# ---------- base interface ----------
class BaseActionConverter:
    """
    Implement both directions:
      - to32(actions_orig)   -> (B,S,32)
      - from32(actions_32)   -> original shape/dim
    """

    def to32(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def from32(self, actions32: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to32_raw_rotation(
        self,
        raw_actions: torch.Tensor,
        *,
        normalized_actions: torch.Tensor | None = None,
        stats: dict[str, Any] | None = None,
        norm_mode: str = "quantile",
    ) -> torch.Tensor:
        """Pack actions with raw YPR rotations and normalized non-rotation dims."""
        del normalized_actions, stats, norm_mode
        raise NotImplementedError(
            f"{type(self).__name__} does not support raw-rotation action encoding"
        )

    def from32_raw_rotation(
        self,
        actions32: torch.Tensor,
        *,
        stats: dict[str, Any] | None = None,
        norm_mode: str = "quantile",
        unnormalize_non_rotation: bool = False,
    ) -> torch.Tensor:
        """Decode actions whose 6D rotation columns represent raw YPR rotations."""
        del stats, norm_mode, unnormalize_non_rotation
        raise NotImplementedError(
            f"{type(self).__name__} does not support raw-rotation action decoding"
        )


# ============================================================
#                     ROBOT CONVERTERS
# ============================================================


class RobotLeftCartesianEuler(BaseActionConverter):
    """
    Input orig: (B,S,7) = [x,y,z,yaw,pitch,roll,grip]
    32-pack:    indices 0..9 = [xyz, R[:,0], R[:,1], grip]
    """

    def to32(self, actions: torch.Tensor) -> torch.Tensor:
        actions = _ensure_bsd(actions)
        if actions.shape[-1] != 7:
            raise ValueError(f"RobotLeft: expected 7-dim, got {actions.shape[-1]}")
        xyz = actions[..., 0:3]
        ypr = actions[..., 3:6]
        g = actions[..., 6:7]
        R = _ypr_to_matrix(ypr)
        c1, c2 = R[..., 0], R[..., 1]
        block = torch.cat([xyz, c1, c2, g], dim=-1)  # (B,S,10)
        return _pad32(block)

    def from32(self, actions32: torch.Tensor) -> torch.Tensor:
        actions32 = _ensure_bsd(actions32)
        block = actions32[..., 0:10]  # left block
        xyz = block[..., 0:3]
        c1 = block[..., 3:6]
        c2 = block[..., 6:9]
        g = block[..., 9:10]
        R = _reconstruct_R_from_cols(c1, c2)
        ypr = _matrix_to_ypr(R)
        return torch.cat([xyz, ypr, g], dim=-1)  # (B,S,7)


class RobotRightCartesianEuler(BaseActionConverter):
    """
    Input orig: (B,S,7)
    32-pack:    indices 10..19 = right block; indices 0..9 are zeros
    """

    def to32(self, actions: torch.Tensor) -> torch.Tensor:
        actions = _ensure_bsd(actions)
        if actions.shape[-1] != 7:
            raise ValueError(f"RobotRight: expected 7-dim, got {actions.shape[-1]}")
        xyz = actions[..., 0:3]
        ypr = actions[..., 3:6]
        g = actions[..., 6:7]
        R = _ypr_to_matrix(ypr)
        c1, c2 = R[..., 0], R[..., 1]
        right = torch.cat([xyz, c1, c2, g], dim=-1)  # (B,S,10)
        zeros = torch.zeros_like(right)
        return _pad32(torch.cat([zeros, right], dim=-1))  # (B,S,20) -> pad 32

    def from32(self, actions32: torch.Tensor) -> torch.Tensor:
        actions32 = _ensure_bsd(actions32)
        block = actions32[..., 10:20]  # right block
        xyz = block[..., 0:3]
        c1 = block[..., 3:6]
        c2 = block[..., 6:9]
        g = block[..., 9:10]
        R = _reconstruct_R_from_cols(c1, c2)
        ypr = _matrix_to_ypr(R)
        return torch.cat([xyz, ypr, g], dim=-1)  # (B,S,7)


class RobotBimanualCartesianEuler(BaseActionConverter):
    """
    Input orig: (B,S,14) = L7 | R7
    32-pack:    left block 0..9, right block 10..19
    """

    def to20(self, actions: torch.Tensor) -> torch.Tensor:
        actions = _ensure_bsd(actions)
        if actions.shape[-1] != 14:
            raise ValueError(f"RobotBimanual: expected 14-dim, got {actions.shape[-1]}")
        L, R = actions[..., :7], actions[..., 7:14]

        # left
        L_xyz, L_ypr, L_g = L[..., 0:3], L[..., 3:6], L[..., 6:7]
        L_R = _ypr_to_matrix(L_ypr)
        L_c1, L_c2 = L_R[..., 0], L_R[..., 1]
        left_block = torch.cat([L_xyz, L_c1, L_c2, L_g], dim=-1)  # (B,S,10)

        # right
        R_xyz, R_ypr, R_g = R[..., 0:3], R[..., 3:6], R[..., 6:7]
        R_R = _ypr_to_matrix(R_ypr)
        R_c1, R_c2 = R_R[..., 0], R_R[..., 1]
        right_block = torch.cat([R_xyz, R_c1, R_c2, R_g], dim=-1)  # (B,S,10)

        return torch.cat([left_block, right_block], dim=-1)  # (B,S,20)

    def to32(self, actions: torch.Tensor) -> torch.Tensor:
        return _pad32(self.to20(actions))

    def from20(self, actions20: torch.Tensor) -> torch.Tensor:
        actions20 = _ensure_bsd(actions20)
        if actions20.shape[-1] < 20:
            raise ValueError(
                f"RobotBimanual: expected at least 20 dims, got {actions20.shape[-1]}"
            )
        Lb = actions20[..., 0:10]
        Rb = actions20[..., 10:20]

        # left
        L_xyz, L_c1, L_c2, L_g = Lb[..., 0:3], Lb[..., 3:6], Lb[..., 6:9], Lb[..., 9:10]
        L_R = _reconstruct_R_from_cols(L_c1, L_c2)
        L_ypr = _matrix_to_ypr(L_R)

        # right
        R_xyz, R_c1, R_c2, R_g = Rb[..., 0:3], Rb[..., 3:6], Rb[..., 6:9], Rb[..., 9:10]
        R_R = _reconstruct_R_from_cols(R_c1, R_c2)
        R_ypr = _matrix_to_ypr(R_R)

        L7 = torch.cat([L_xyz, L_ypr, L_g], dim=-1)
        R7 = torch.cat([R_xyz, R_ypr, R_g], dim=-1)
        return torch.cat([L7, R7], dim=-1)  # (B,S,14)

    def from32(self, actions32: torch.Tensor) -> torch.Tensor:
        return self.from20(actions32)

    def to20_raw_rotation(
        self,
        raw_actions: torch.Tensor,
        *,
        normalized_actions: torch.Tensor | None = None,
        stats: dict[str, Any] | None = None,
        norm_mode: str = "quantile",
    ) -> torch.Tensor:
        raw_actions = _ensure_bsd(raw_actions)
        if raw_actions.shape[-1] != 14:
            raise ValueError(
                f"RobotBimanual: expected 14-dim, got {raw_actions.shape[-1]}"
            )
        if normalized_actions is None:
            if stats is None:
                raise ValueError("stats are required when normalized_actions is omitted")
            model_actions = _normalize_robot_bimanual_non_rot(
                raw_actions, stats, norm_mode
            )
        else:
            normalized_actions = _ensure_bsd(normalized_actions).to(raw_actions.device)
            if normalized_actions.shape != raw_actions.shape:
                raise ValueError(
                    "normalized_actions must match raw_actions shape; got "
                    f"{tuple(normalized_actions.shape)} vs {tuple(raw_actions.shape)}"
                )
            model_actions = raw_actions.clone()
            model_actions[..., ROBOT_BIMANUAL_CARTESIAN_NON_ROT_DIMS] = (
                normalized_actions[..., ROBOT_BIMANUAL_CARTESIAN_NON_ROT_DIMS]
            )
        return self.to20(model_actions)

    def from20_raw_rotation(
        self,
        actions20: torch.Tensor,
        *,
        stats: dict[str, Any] | None = None,
        norm_mode: str = "quantile",
        unnormalize_non_rotation: bool = False,
    ) -> torch.Tensor:
        model_actions = self.from20(actions20)
        if not unnormalize_non_rotation:
            return model_actions
        if stats is None:
            raise ValueError("stats are required to unnormalize non-rotation dims")
        return _unnormalize_robot_bimanual_non_rot(model_actions, stats, norm_mode)

    def to32_raw_rotation(
        self,
        raw_actions: torch.Tensor,
        *,
        normalized_actions: torch.Tensor | None = None,
        stats: dict[str, Any] | None = None,
        norm_mode: str = "quantile",
    ) -> torch.Tensor:
        return _pad32(
            self.to20_raw_rotation(
                raw_actions,
                normalized_actions=normalized_actions,
                stats=stats,
                norm_mode=norm_mode,
            )
        )

    def from32_raw_rotation(
        self,
        actions32: torch.Tensor,
        *,
        stats: dict[str, Any] | None = None,
        norm_mode: str = "quantile",
        unnormalize_non_rotation: bool = False,
    ) -> torch.Tensor:
        return self.from20_raw_rotation(
            actions32,
            stats=stats,
            norm_mode=norm_mode,
            unnormalize_non_rotation=unnormalize_non_rotation,
        )


# ============================================================
#                     HUMAN CONVERTERS
# ============================================================


class HumanLeftCartesianEuler(BaseActionConverter):
    """
    Input orig: (B,S,6) = [x,y,z,yaw,pitch,roll]
    32-pack:    left block 0..9 with g=0
    """

    def to32(self, actions: torch.Tensor) -> torch.Tensor:
        actions = _ensure_bsd(actions)
        if actions.shape[-1] != 6:
            raise ValueError(f"HumanLeft: expected 6-dim, got {actions.shape[-1]}")
        xyz, ypr = actions[..., 0:3], actions[..., 3:6]
        R = _ypr_to_matrix(ypr)
        c1, c2 = R[..., 0], R[..., 1]
        g0 = torch.zeros_like(xyz[..., :1])
        block = torch.cat([xyz, c1, c2, g0], dim=-1)  # (B,S,10)
        return _pad32(block)

    def from32(self, actions32: torch.Tensor) -> torch.Tensor:
        actions32 = _ensure_bsd(actions32)
        block = actions32[..., 0:10]
        xyz, c1, c2 = block[..., 0:3], block[..., 3:6], block[..., 6:9]
        R = _reconstruct_R_from_cols(c1, c2)
        ypr = _matrix_to_ypr(R)
        return torch.cat([xyz, ypr], dim=-1)  # (B,S,6)


class HumanRightCartesianEuler(BaseActionConverter):
    """
    Input orig: (B,S,6)
    32-pack:    zeros 0..9, right block 10..19 with g=0 in block
    """

    def to32(self, actions: torch.Tensor) -> torch.Tensor:
        actions = _ensure_bsd(actions)
        if actions.shape[-1] != 6:
            raise ValueError(f"HumanRight: expected 6-dim, got {actions.shape[-1]}")
        xyz, ypr = actions[..., 0:3], actions[..., 3:6]
        R = _ypr_to_matrix(ypr)
        c1, c2 = R[..., 0], R[..., 1]
        g0 = torch.zeros_like(xyz[..., :1])
        block = torch.cat([xyz, c1, c2, g0], dim=-1)  # (B,S,10)
        zeros = torch.zeros_like(block)
        return _pad32(torch.cat([zeros, block], dim=-1))

    def from32(self, actions32: torch.Tensor) -> torch.Tensor:
        actions32 = _ensure_bsd(actions32)
        block = actions32[..., 10:20]
        xyz, c1, c2 = block[..., 0:3], block[..., 3:6], block[..., 6:9]
        R = _reconstruct_R_from_cols(c1, c2)
        ypr = _matrix_to_ypr(R)
        return torch.cat([xyz, ypr], dim=-1)  # (B,S,6)


class HumanBimanualCartesianEuler(BaseActionConverter):
    """
    Input orig: (B,S,12) = L6 | R6
    32-pack:    left 0..9 (g=0), right 10..19 (g=0)
    """

    def to32(self, actions: torch.Tensor) -> torch.Tensor:
        actions = _ensure_bsd(actions)
        if actions.shape[-1] != 12:
            raise ValueError(f"HumanBimanual: expected 12-dim, got {actions.shape[-1]}")
        L, R = actions[..., :6], actions[..., 6:12]

        L_xyz, L_ypr = L[..., 0:3], L[..., 3:6]
        L_R = _ypr_to_matrix(L_ypr)
        L_c1, L_c2 = L_R[..., 0], L_R[..., 1]
        g0L = torch.zeros_like(L_xyz[..., :1])
        Lblock = torch.cat([L_xyz, L_c1, L_c2, g0L], dim=-1)  # (B,S,10)

        R_xyz, R_ypr = R[..., 0:3], R[..., 3:6]
        R_R = _ypr_to_matrix(R_ypr)
        R_c1, R_c2 = R_R[..., 0], R_R[..., 1]
        g0R = torch.zeros_like(R_xyz[..., :1])
        Rblock = torch.cat([R_xyz, R_c1, R_c2, g0R], dim=-1)  # (B,S,10)

        return _pad32(torch.cat([Lblock, Rblock], dim=-1))

    def from32(self, actions32: torch.Tensor) -> torch.Tensor:
        actions32 = _ensure_bsd(actions32)
        Lb = actions32[..., 0:10]
        Rb = actions32[..., 10:20]
        L_xyz, L_c1, L_c2 = Lb[..., 0:3], Lb[..., 3:6], Lb[..., 6:9]
        R_xyz, R_c1, R_c2 = Rb[..., 0:3], Rb[..., 3:6], Rb[..., 6:9]
        L_R = _reconstruct_R_from_cols(L_c1, L_c2)
        L_ypr = _matrix_to_ypr(L_R)
        R_R = _reconstruct_R_from_cols(R_c1, R_c2)
        R_ypr = _matrix_to_ypr(R_R)
        return torch.cat([L_xyz, L_ypr, R_xyz, R_ypr], dim=-1)  # (B,S,12)
