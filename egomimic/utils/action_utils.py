"""pi0.5 action I/O.

Actions arrive from the data pipeline already in their model-native layout
and already normalized by the standard MultiDataset pipeline:

- cartesian: xyz+rot6d(+gripper) per arm (the ypr->6D conversion is the
  ``CartesianYPRToRot6D`` data transform, norm stats are computed on 6D);
- hand keypoints: the 144-D wrist-first ``[wrist xyz | wrist rot6d |
  21 keypoints] x 2`` vector (``keypoints_*_6d`` modes).

The forward pass only *packs* the normalized vector into openpi's action
vector (``to32_norm_6d`` builds the canonical 32-slot block layout, and the
PI algo pads that to ``model.action_dim``) and the eval path unpacks it
(``from32_norm_6d``). No rotation math and no normalization happen here.
"""

from typing import Any, Dict, Sequence, Tuple

import torch


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


def pad_to_width(x: torch.Tensor, width: int) -> torch.Tensor:
    """Zero-pad the last dim of a (B,S,D) tensor up to ``width``.

    Raises when ``D > width``: a wider native action than the model's
    ``action_dim`` would otherwise be truncated silently.
    """
    x = _ensure_bsd(x)
    B, S, D = x.shape
    if D == width:
        return x
    if D > width:
        raise ValueError(
            f"Action has {D} dims but the model action_dim is {width}; set "
            "model.robomimic_model.config.model.action_dim >= the widest "
            "embodiment action"
        )
    pad = torch.zeros(B, S, width - D, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def _pad32(x: torch.Tensor) -> torch.Tensor:
    return pad_to_width(x, 32)


def _stat_tensor(stats: dict[str, Any], key: str, ref: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(stats[key], device=ref.device, dtype=torch.float32)
    return value.to(dtype=ref.dtype if ref.is_floating_point() else torch.float32)


# A channel whose stat range (std, max - min, or q99 - q1) is below this is
# treated as constant: it normalizes to 0 and unnormalizes to its centre.
# Structural zeros, not rare accidents: the t=0 cell of a wrist-frame action
# chunk is exactly the identity pose, and in a wrist-frame keypoint proprio the
# wrist keypoint sits at its own frame origin while kp9 defines the forward
# axis, so kp0 and kp9's off-axis components are zero at EVERY timestep (28
# action cells and 10 proprio channels on a real 144-D episode). A bare
# ``+ 1e-6`` denominator scaled any off-convention value in them by 1e6.
# One absolute threshold across channels with different units is deliberate:
# 1e-4 is 0.1 mm of translation but 1/20000 of a rot6d channel's range. Same
# rule and the same constant as Diffusion Policy / robomimic; GR00T instead
# clamps the range, which keeps dividing.
NORM_MIN_RANGE = 1e-4


def _norm_center_scale(
    stats: dict[str, Any], norm_mode: str, ref: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(center, half_range)`` so that ``normalized = (x - center) / half_range``
    (zscore: ``(mean, std)``)."""
    if norm_mode == "zscore":
        return _stat_tensor(stats, "mean", ref), _stat_tensor(stats, "std", ref)
    if norm_mode == "minmax":
        lo, hi = _stat_tensor(stats, "min", ref), _stat_tensor(stats, "max", ref)
    elif norm_mode == "quantile":
        lo = _stat_tensor(stats, "quantile_1", ref)
        hi = _stat_tensor(stats, "quantile_99", ref)
    else:
        raise ValueError(f"Invalid normalization mode: {norm_mode}")
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def _eps(norm_mode: str) -> float:
    # Keeps the historical denominators: std + 1e-6 and (hi - lo) + 1e-6.
    return 1e-6 if norm_mode == "zscore" else 0.5e-6


def _degenerate(scale: torch.Tensor, norm_mode: str) -> torch.Tensor:
    full_range = scale if norm_mode == "zscore" else 2.0 * scale
    return full_range < NORM_MIN_RANGE


def _passthrough(
    out: torch.Tensor, tensor: torch.Tensor, channels: Sequence[int] | None
) -> torch.Tensor:
    if not channels:
        return out
    out = out.clone()
    out[..., list(channels)] = tensor[..., list(channels)]
    return out


def _apply_norm_one(
    tensor: torch.Tensor,
    stats: dict[str, Any],
    norm_mode: str,
    identity_channels: Sequence[int] | None = None,
) -> torch.Tensor:
    """Normalize with per-key stats (zscore, or [-1, 1] for minmax/quantile).
    Channels with a range below ``NORM_MIN_RANGE`` map to 0;
    ``identity_channels`` pass through untouched."""
    center, scale = _norm_center_scale(stats, norm_mode, tensor)
    out = (tensor - center) / (scale + _eps(norm_mode))
    out = torch.where(_degenerate(scale, norm_mode), torch.zeros_like(out), out)
    return _passthrough(out, tensor, identity_channels)


def _apply_unnorm_one(
    tensor: torch.Tensor,
    stats: dict[str, Any],
    norm_mode: str,
    identity_channels: Sequence[int] | None = None,
) -> torch.Tensor:
    """Inverse of :func:`_apply_norm_one`; degenerate channels return their
    centre (mean / midpoint) whatever the model predicted."""
    center, scale = _norm_center_scale(stats, norm_mode, tensor)
    out = tensor * (scale + _eps(norm_mode)) + center
    out = torch.where(_degenerate(scale, norm_mode), center.expand_as(out), out)
    return _passthrough(out, tensor, identity_channels)


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
    Pack / unpack between an embodiment's native normalized action and the
    pi0.5 action vector:
      - to32_norm_6d(actions)     -> (B,S,>=32) canonical block layout
      - from32_norm_6d(actions32) -> native shape/dim
    The base class implements neither; it is the ``fallback`` converter in the
    model yamls and raises so an unregistered embodiment fails loudly.
    """

    def to32_norm_6d(self, actions: torch.Tensor) -> torch.Tensor:
        """Pack an already-normalized native action into the canonical
        pi0.5 layout (pure rearrange: no rotation math, no normalization)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support normalized-rot6d encoding"
        )

    def from32_norm_6d(self, actions32: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`to32_norm_6d`: extract the normalized native
        action from the model's action vector (pure rearrange)."""
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


class HumanBimanualKeypoints(BaseActionConverter):
    """144-D wrist-first MANO keypoint action, per hand
    ``[wrist xyz+rot6d in its own wrist frame (9) | 21 keypoints in that
    wrist frame (63)]`` x {left, right} (``keypoints_wristframe_6d``), packed
    identity-first into the model's action vector. The PI algo resizes
    openpi's action projections to ``model.action_dim`` (>= 144) and pads the
    vector up to it; ``from32_norm_6d`` slices the native width back out.
    """

    native_dim = 144

    def to32_norm_6d(self, actions: torch.Tensor) -> torch.Tensor:
        actions = _ensure_bsd(actions)
        if actions.shape[-1] != self.native_dim:
            raise ValueError(
                f"HumanBimanualKeypoints: expected {self.native_dim}-dim, got "
                f"{actions.shape[-1]}"
            )
        return actions

    def from32_norm_6d(self, actions32: torch.Tensor) -> torch.Tensor:
        actions32 = _ensure_bsd(actions32)
        if actions32.shape[-1] < self.native_dim:
            raise ValueError(
                f"HumanBimanualKeypoints: expected >={self.native_dim} dims, got "
                f"{actions32.shape[-1]}"
            )
        return actions32[..., : self.native_dim]
