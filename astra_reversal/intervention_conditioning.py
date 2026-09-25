"""Bounded changes to native language embeddings before prefix-cache creation.

This operator preserves token positions and padding. It does not promise that
an edited embedding preserves task meaning or improves the policy's behavior.
"""

import math
from dataclasses import dataclass
from numbers import Real


@dataclass(frozen=True)
class TextEmbeddingIntervention:
    guidance_prompt: str
    alpha: float
    max_relative_norm: float = 0.25

    def __post_init__(self):
        if (
            not isinstance(self.guidance_prompt, str)
            or not self.guidance_prompt.strip()
        ):
            raise ValueError("guidance_prompt must be a nonempty string")
        for name, lower, upper in (
            ("alpha", 0.0, 1.0),
            ("max_relative_norm", 0.0, 0.25),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
            object.__setattr__(self, name, float(value))


def pooled_text_residual(
    original,
    original_mask,
    guidance,
    guidance_mask,
    *,
    alpha,
    max_relative_norm=0.25,
):
    """Return edited native text embeddings and measured single-condition norms.

    The direction is the difference of valid-token means. The scaled direction
    is added to each originally valid text slot; no slot is created or removed.
    Float64 accumulators compute means/norms, and the result retains the native
    embedding dtype. The bound applies to the *stored* change after rounding.
    """
    import torch

    spec = TextEmbeddingIntervention(
        "validated outside this numeric helper", alpha, max_relative_norm
    )

    def validate(values, mask, name):
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 3
            or values.shape[0] != 1
            or values.shape[1] < 1
            or values.shape[2] < 1
            or not values.is_floating_point()
            or not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != values.shape[:2]
            or mask.device != values.device
        ):
            raise ValueError(
                f"{name} must be finite [1, tokens, width] embeddings with a boolean token mask"
            )
        if not torch.isfinite(values).all().item() or not mask.any().item():
            raise ValueError(
                f"{name} must have finite embeddings and at least one valid token"
            )

    validate(original, original_mask, "Original text")
    with torch.no_grad():
        base = original[original_mask].to(torch.float64)
        base_norm = torch.linalg.vector_norm(base).item()
        if not math.isfinite(base_norm):
            raise ValueError("Original text norm must be finite")
        stats = {
            "original_text_frobenius": base_norm,
            "requested_delta_frobenius": 0.0,
            "delta_frobenius": 0.0,
            "relative_rms": 0.0,
            "delta_rms": 0.0,
            "bound_frobenius": spec.max_relative_norm * base_norm,
            "bound_applied": False,
            "rounding_bound_correction": False,
            "precision_limited_to_identity": False,
            "effective_direction_scale": 0.0,
            "accumulator_dtype": "float64",
        }
        if spec.alpha == 0.0 or spec.max_relative_norm == 0.0:
            # Do not inspect or tokenize guidance for a disabled intervention.
            return original, stats
        validate(guidance, guidance_mask, "Guidance text")
        if (
            guidance.shape[2] != original.shape[2]
            or guidance.device != original.device
            or guidance.dtype != original.dtype
        ):
            raise ValueError(
                "Guidance and original embedding width/device/dtype must match"
            )
        direction = guidance[guidance_mask].to(torch.float64).mean(0) - base.mean(0)
        requested_norm = (
            torch.linalg.vector_norm(direction).item()
            * math.sqrt(base.shape[0])
            * spec.alpha
        )
        if not math.isfinite(requested_norm):
            raise ValueError("Requested text residual norm must be finite")
        bound = stats["bound_frobenius"]
        coefficient = spec.alpha
        if requested_norm > bound:
            coefficient *= bound / requested_norm
        stats.update(
            requested_delta_frobenius=requested_norm,
            bound_applied=requested_norm > bound,
        )

        # Rounding an update can slightly exceed its ideal bound. Recheck the
        # actual stored tensor, reducing the coefficient only when necessary.
        for attempt in range(5):
            edited = original.clone()
            edited[original_mask] = (base + coefficient * direction).to(original.dtype)
            actual_norm = torch.linalg.vector_norm(
                edited[original_mask].to(torch.float64) - base
            ).item()
            if math.isfinite(actual_norm) and actual_norm <= bound:
                break
            stats["rounding_bound_correction"] = True
            if attempt == 4 or not math.isfinite(actual_norm):
                edited, actual_norm, coefficient = original, 0.0, 0.0
                stats["precision_limited_to_identity"] = True
                break
            coefficient *= (bound / actual_norm) * (
                1 - 32 * torch.finfo(original.dtype).eps
            )

        stats.update(
            delta_frobenius=actual_norm,
            relative_rms=actual_norm / base_norm if base_norm else 0.0,
            delta_rms=actual_norm / math.sqrt(base.numel()),
            effective_direction_scale=coefficient,
        )
        return edited, stats
