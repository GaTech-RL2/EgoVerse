"""Bounded, recorded interventions; no model weights or evaluation goals change."""

import copy

import numpy as np

from .records import digest

ARMS = (
    "random_noise",
    "noise_only",
    "language_only",
    "vision_only",
    "noise_language",
    "noise_vision",
    "language_vision",
    "joint",
)
CAMERAS = ("observation/image", "observation/wrist_image")


def noise_basis(shape, seed, rank=8):
    """An orthogonal basis with unit RMS vectors in the *full* flow state.

    Coefficients have Euclidean norm <= 1, hence perturbation_scale bounds the
    RMS noise change. These axes have no assumed physical action semantics.
    """
    size = int(np.prod(shape))
    if len(shape) != 3 or shape[0] != 1 or not 1 <= rank <= size:
        raise ValueError("Expected a single full flow state and a valid rank")
    rng = np.random.default_rng(seed)
    orthogonal, _ = np.linalg.qr(rng.standard_normal((size, rank)))
    basis = (orthogonal.T * np.sqrt(size)).reshape((rank, *shape)).astype(np.float32)
    return basis, digest(basis)


def perturb_noise(recovered, basis, proposal):
    recovered = np.asarray(recovered, dtype=np.float32)
    if proposal is None:
        return recovered.copy()
    if proposal.get("basis_id") != digest(basis):
        raise ValueError("Noise proposal belongs to a different basis")
    coefficients = np.asarray(proposal["coefficients"], dtype=np.float64)
    scale = proposal["perturbation_scale"]
    if (
        coefficients.shape != (len(basis),)
        or not np.isfinite(coefficients).all()
        or np.linalg.norm(coefficients) > 1 + 1e-7
        or type(scale) not in (int, float)
        or not np.isfinite(scale)
        or not 0 <= scale <= 0.5
        or basis.shape[1:] != recovered.shape
    ):
        raise ValueError("Invalid bounded noise intervention")
    delta = np.tensordot(coefficients, basis, axes=1).astype(np.float32) * scale
    result = recovered + delta
    if not np.isfinite(result).all():
        raise ValueError("Noise intervention became nonfinite")
    return result


def random_noise_proposal(basis, rng):
    coefficients = rng.standard_normal(len(basis))
    coefficients /= np.linalg.norm(coefficients)
    return {
        "kind": "low_rank",
        "basis_id": digest(basis),
        "coefficients": coefficients.tolist(),
        "perturbation_scale": float(rng.choice([0.15, 0.3, 0.5])),
    }


def apply_vision(observation, annotations):
    """Render static, translucent magenta marks without changing image geometry.

    Marks persist for this candidate, in the original camera coordinates. There
    is no object tracker. Feedback to Astra always uses the unmodified frames.
    """
    from PIL import Image, ImageDraw

    result = copy.deepcopy(observation)
    for annotation in annotations:
        camera = annotation["camera"]
        if camera not in CAMERAS:
            raise ValueError("Unknown vision camera")
        pixels = np.asarray(result[camera])
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("Interventions require uint8 RGB camera frames")
        gain = annotation["gain"]
        if (
            type(gain) not in (int, float)
            or not np.isfinite(gain)
            or not 0 <= gain <= 1
        ):
            raise ValueError("Vision gain must be in [0, 1]")
        coordinates = np.asarray(annotation["coordinates"], dtype=np.float64)
        kind = annotation["kind"]
        if kind not in ("point", "box") or coordinates.shape != (
            2 if kind == "point" else 4,
        ):
            raise ValueError("Invalid annotation geometry")
        if not np.isfinite(coordinates).all():
            raise ValueError("Nonfinite annotation")
        if any(
            not 0 <= value < pixels.shape[1 - i % 2]
            for i, value in enumerate(coordinates)
        ):
            raise ValueError("Annotation is outside the image")
        if kind == "box" and (
            coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]
        ):
            raise ValueError("Box corners are not ordered")
        if gain == 0:
            continue
        image = Image.fromarray(pixels)
        layer = image.copy()
        draw = ImageDraw.Draw(layer)
        xy = coordinates.tolist()
        if kind == "box":
            draw.rectangle(xy, outline=(255, 0, 255), width=3)
        else:
            x, y = xy
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 255))
        result[camera] = np.array(Image.blend(image, layer, float(gain)), copy=True)
    return result


def success_curve(attempts, budget=5):
    """Include API failures in attempt count; never treat censoring as success."""
    first = next((row["iteration"] for row in attempts if row["success"]), None)
    return {
        "first_success_attempt": first,
        "intervention_iterations_to_success": None if first is None else first - 1,
        "censored_after_attempt": None if first is not None else budget,
        "success_by_attempt": [
            first is not None and first <= i for i in range(1, budget + 1)
        ],
        "candidate_rollouts": sum(
            bool(row.get("rollout_executed")) for row in attempts
        ),
        "proposal_failures": sum(
            row.get("status") == "proposal_error" for row in attempts
        ),
    }
