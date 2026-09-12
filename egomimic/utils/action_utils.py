from typing import Any, Dict, Tuple

import torch

# PI0.5 action I/O. Actions arrive from the data pipeline already in the
# xyz+6D(+gripper) layout (the ypr->6D conversion is done by the
# ``CartesianYPRToRot6D`` data transform, and the norm stats are computed on
# the 6D data) and already normalized by the standard MultiDataset pipeline.
# The forward pass only *packs* the normalized 6D action into openpi's 32D
# vector (``to32_norm_6d``) and the eval path unpacks it (``from32_norm_6d``).
# No rotation math and no normalization happen in the model.


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


# ---------- shared helpers ----------
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
    """Tensor twin of the MultiDataset normalization formula (used by tests)."""
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
    """Inverse of :func:`_apply_norm_one`."""
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


# ---------- rotation helpers (shared with the eval metrics) ----------
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
    Pack / unpack between an embodiment's native normalized xyz+6D(+gripper)
    action and openpi's 32D action vector:
      - to32_norm_6d(actions)     -> (B,S,32)
      - from32_norm_6d(actions32) -> native shape/dim
    The base class implements neither; it is the ``fallback`` converter in the
    model yamls and raises so an unregistered embodiment fails loudly.
    """

    def to32_norm_6d(self, actions: torch.Tensor) -> torch.Tensor:
        """Pack an already-normalized xyz+6D(+gripper) action into the 32D vector.

        The ypr->6D conversion happens upstream in the ``CartesianYPRToRot6D``
        data transform and the result is normalized by the standard data
        pipeline, so this is a pure rearrange (no rotation math, no
        normalization).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support normalized-rot6d encoding"
        )

    def from32_norm_6d(self, actions32: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`to32_norm_6d`: extract the normalized xyz+6D(+gripper)
        action from the 32D vector (pure rearrange)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support normalized-rot6d decoding"
        )


# ============================================================
#                     ROBOT CONVERTERS
# ============================================================


class RobotBimanualCartesianEuler(BaseActionConverter):
    """
    Native: (B,S,20) = [L xyz(3) 6d(6) g(1) | R xyz(3) 6d(6) g(1)]
    32-pack: left block 0..9, right block 10..19, zero pad 20..31
    """

    def to32_norm_6d(self, actions: torch.Tensor) -> torch.Tensor:
        # The native 20D layout already IS the canonical 32D block layout
        # (left 0..9, right 10..19), just pad.
        actions = _ensure_bsd(actions)
        if actions.shape[-1] != 20:
            raise ValueError(
                f"RobotBimanual.to32_norm_6d expected 20-dim, got {actions.shape[-1]}"
            )
        return _pad32(actions)

    def from32_norm_6d(self, actions32: torch.Tensor) -> torch.Tensor:
        actions32 = _ensure_bsd(actions32)
        if actions32.shape[-1] < 20:
            raise ValueError(
                f"RobotBimanual.from32_norm_6d expected >=20 dims, got "
                f"{actions32.shape[-1]}"
            )
        return actions32[..., 0:20]


# ============================================================
#                     HUMAN CONVERTERS
# ============================================================


class HumanBimanualCartesianEuler(BaseActionConverter):
    """
    Native: (B,S,18) = [L xyz(3) 6d(6) | R xyz(3) 6d(6)]  (no gripper)
    32-pack: left block 0..9 (g=0), right block 10..19 (g=0), zero pad 20..31
    """

    def to32_norm_6d(self, actions: torch.Tensor) -> torch.Tensor:
        # Human has no gripper, so insert a zero gripper slot at the end of
        # each arm block to match the 32D block layout [xyz(3) c1(3) c2(3) g(1)] x 2.
        actions = _ensure_bsd(actions)
        if actions.shape[-1] != 18:
            raise ValueError(
                f"HumanBimanual.to32_norm_6d expected 18-dim, got {actions.shape[-1]}"
            )
        L = actions[..., 0:9]
        R = actions[..., 9:18]
        g0 = torch.zeros_like(actions[..., :1])
        Lblock = torch.cat([L, g0], dim=-1)  # (B,S,10)
        Rblock = torch.cat([R, g0], dim=-1)  # (B,S,10)
        return _pad32(torch.cat([Lblock, Rblock], dim=-1))

    def from32_norm_6d(self, actions32: torch.Tensor) -> torch.Tensor:
        actions32 = _ensure_bsd(actions32)
        if actions32.shape[-1] < 20:
            raise ValueError(
                f"HumanBimanual.from32_norm_6d expected >=20 dims, got "
                f"{actions32.shape[-1]}"
            )
        L = actions32[..., 0:9]  # drop left gripper slot at idx 9
        R = actions32[..., 10:19]  # drop right gripper slot at idx 19
        return torch.cat([L, R], dim=-1)  # (B,S,18)
