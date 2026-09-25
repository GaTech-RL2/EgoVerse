"""Stateless, auditable perturbations of raw uint8 RGB policy observations.

``demo_blend`` uses an immutable same-camera demonstration image. ``occlusion``
blends a constant RGB fill inside an integer, half-open rectangle. Neither
operator changes image geometry, rotates/resizes frames, or edits other fields.
The caller owns decision lifetime: apply an active specification to each fresh
raw observation, never to an image returned by a previous application.
"""

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .records import digest

SCHEMA_VERSION = "astra-image-perturbation-1.0"
CAMERAS = ("observation/image", "observation/wrist_image")
KINDS = ("demo_blend", "occlusion")
ROUNDING = "float64 separate convex terms; clip[0,255]; floor(value+0.5); uint8"


def _unit(value, name):
    if (
        type(value) not in (int, float)
        or not 0 <= value <= 1
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def _rgb(value, name):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(channel) is not int or not 0 <= channel <= 255 for channel in value)
    ):
        raise ValueError(f"{name} must contain three integer RGB values in [0, 255]")
    return tuple(value)


@dataclass(frozen=True)
class ImagePerturbationLimits:
    """Frozen experiment bounds; violations are rejected, never clipped to fit."""

    allowed_kinds: tuple[str, ...] = KINDS
    max_operations: int = 2
    max_alpha: float = 1.0
    max_strength: float = 1.0
    max_occlusion_fraction: float = 0.5
    fixed_fill_rgb: tuple[int, int, int] = (127, 127, 127)

    def __post_init__(self):
        if (
            type(self.allowed_kinds) is not tuple
            or any(
                type(kind) is not str or kind not in KINDS
                for kind in self.allowed_kinds
            )
            or len(set(self.allowed_kinds)) != len(self.allowed_kinds)
        ):
            raise ValueError("allowed_kinds must be a unique tuple of supported kinds")
        if type(self.max_operations) is not int or not 0 <= self.max_operations <= 2:
            raise ValueError("max_operations must be an integer in [0, 2]")
        for name in ("max_alpha", "max_strength", "max_occlusion_fraction"):
            _unit(getattr(self, name), name)
        if type(self.fixed_fill_rgb) is not tuple:
            raise ValueError("fixed_fill_rgb must be an immutable RGB tuple")
        _rgb(self.fixed_fill_rgb, "fixed_fill_rgb")

    def as_dict(self):
        return {
            "allowed_kinds": list(self.allowed_kinds),
            "max_operations": self.max_operations,
            "max_alpha": float(self.max_alpha),
            "max_strength": float(self.max_strength),
            "max_occlusion_fraction": float(self.max_occlusion_fraction),
            "fixed_fill_rgb": list(self.fixed_fill_rgb),
        }


DEFAULT_LIMITS = ImagePerturbationLimits()


def _shape(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(size) is not int or size <= 0 for size in value)
        or value[2] != 3
    ):
        raise ValueError("Camera shapes must be positive integer (height, width, 3)")
    return tuple(value)


def _identifier(value):
    return type(value) is str and bool(value) and value.strip() == value


def validate_image_perturbations(
    operations, image_shapes, *, donor_ids=None, limits=DEFAULT_LIMITS
):
    """Validate exact JSON keys/types, cameras, bounds and optional donor catalog.

    ``image_shapes`` maps both canonical cameras to their raw ``(H, W, 3)``
    shapes. ``donor_ids`` optionally restricts selected paired-sample IDs. The
    applying function additionally resolves the selected camera and verifies its
    actual pixels against immutable donor provenance, including for alpha zero.
    """
    if not isinstance(limits, ImagePerturbationLimits):
        raise ValueError("limits must be ImagePerturbationLimits")
    if not isinstance(image_shapes, Mapping) or any(
        camera not in image_shapes for camera in CAMERAS
    ):
        raise ValueError("Both canonical camera shapes are required")
    shapes = {camera: _shape(image_shapes[camera]) for camera in CAMERAS}
    if type(operations) is not list or len(operations) > limits.max_operations:
        raise ValueError("Operations must be a list within the operation-count limit")
    if donor_ids is not None:
        if isinstance(donor_ids, (str, bytes)):
            raise ValueError("donor_ids must be a collection of IDs")
        try:
            donor_ids = list(donor_ids)
        except TypeError:
            raise ValueError("donor_ids must be a collection of IDs") from None
        if any(not _identifier(value) for value in donor_ids):
            raise ValueError("Donor catalog contains an invalid ID")
        donor_ids = set(donor_ids)
    seen = set()
    for operation in operations:
        if type(operation) is not dict:
            raise ValueError("Each image perturbation must be a JSON object")
        kind = operation.get("kind")
        camera = operation.get("camera")
        if type(kind) is not str or kind not in limits.allowed_kinds:
            raise ValueError("Image perturbation kind is not allowed by this arm")
        if type(camera) is not str or camera not in CAMERAS:
            raise ValueError("Unknown image perturbation camera")
        if camera in seen:
            raise ValueError("At most one operation is allowed per camera")
        seen.add(camera)
        if kind == "demo_blend":
            if set(operation) != {"kind", "camera", "donor_id", "alpha"}:
                raise ValueError(
                    "demo_blend requires exactly kind, camera, donor_id, alpha"
                )
            if not _identifier(operation["donor_id"]):
                raise ValueError("Invalid donor_id")
            if donor_ids is not None and operation["donor_id"] not in donor_ids:
                raise ValueError("Selected donor_id is absent from the pinned catalog")
            if _unit(operation["alpha"], "alpha") > limits.max_alpha:
                raise ValueError("alpha exceeds the experiment limit")
        else:
            if set(operation) != {"kind", "camera", "box_xyxy", "fill_rgb", "strength"}:
                raise ValueError(
                    "occlusion requires exactly kind, camera, box_xyxy, fill_rgb, strength"
                )
            box = operation["box_xyxy"]
            if (
                type(box) is not list
                or len(box) != 4
                or any(type(value) is not int for value in box)
            ):
                raise ValueError(
                    "box_xyxy must be four integer half-open pixel coordinates"
                )
            x0, y0, x1, y1 = box
            height, width, _ = shapes[camera]
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(
                    "Occlusion rectangle must be nonempty and inside its camera"
                )
            if (x1 - x0) * (y1 - y0) / (width * height) > limits.max_occlusion_fraction:
                raise ValueError("Occlusion area exceeds the experiment limit")
            if (
                type(operation["fill_rgb"]) is not list
                or _rgb(operation["fill_rgb"], "fill_rgb") != limits.fixed_fill_rgb
            ):
                raise ValueError(
                    "fill_rgb must equal the experiment's fixed neutral RGB"
                )
            if _unit(operation["strength"], "strength") > limits.max_strength:
                raise ValueError("strength exceeds the experiment limit")
    return copy.deepcopy(operations)


def _pixels(value, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ValueError(f"{name} must be a uint8 RGB ndarray")
    _shape(value.shape)
    return np.array(value, copy=True, order="C")


def _sha(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _donor(donor_lookup, donor_id, camera, shape):
    if not callable(donor_lookup):
        raise ValueError("demo_blend requires a pinned same-camera donor lookup")
    try:
        donor = donor_lookup(donor_id, camera)
    except KeyError:
        raise ValueError(
            "Selected donor/camera is absent from the pinned library"
        ) from None
    if (
        getattr(donor, "donor_id", None) != donor_id
        or getattr(donor, "camera", None) != camera
    ):
        raise ValueError("Resolved donor ID/camera does not match the requested pair")
    pixels = _pixels(getattr(donor, "pixels", None), "Donor pixels")
    if pixels.shape != shape:
        raise ValueError(
            "Donor and raw camera must have identical shapes; implicit warping is forbidden"
        )
    provenance = copy.deepcopy(getattr(donor, "provenance", None))
    if type(provenance) is not dict or any(
        not _sha(provenance.get(key))
        for key in ("library_id", "sample_sha256", "pixels_sha256")
    ):
        raise ValueError(
            "Donor provenance must pin library, sample and pixel SHA256 values"
        )
    try:
        json.dumps(provenance, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("Donor provenance must be finite JSON data") from None
    if provenance["pixels_sha256"] != digest(pixels):
        raise ValueError("Donor pixels do not match their pinned digest")
    return pixels, provenance


def _convex_uint8(original, replacement, weight):
    """Separate float64 multiplies/add, then nonnegative round-half-up."""
    weight = float(weight)
    original_term = original.astype(np.float64) * (1.0 - weight)
    replacement_term = np.asarray(replacement, dtype=np.float64) * weight
    mixture = original_term + replacement_term
    return np.floor(np.clip(mixture, 0.0, 255.0) + 0.5).astype(np.uint8)


def _differences(before, after):
    delta = after.astype(np.int16) - before.astype(np.int16)
    changed = int(np.count_nonzero(np.any(delta != 0, axis=2)))
    square_sum = int(np.sum(delta.astype(np.int64) ** 2, dtype=np.int64))
    return {
        "changed_pixels": changed,
        "changed_fraction": changed / (before.shape[0] * before.shape[1]),
        "changed_channel_values": int(np.count_nonzero(delta)),
        "rms": math.sqrt(square_sum / delta.size),
        "linf": int(np.max(np.abs(delta))),
        "metric_units": "uint8_levels_0_255",
    }


def apply_image_perturbations(
    raw, operations, donor_lookup=None, *, limits=DEFAULT_LIMITS
):
    """Return ``(new_observation, audit)`` without mutating inputs or retaining state.

    Zero weights and an empty list preserve every observation byte. Metadata
    reports requested mask area separately from actual changed pixel fraction:
    a valid nonzero edit can quantize to a no-op or match existing pixel values.
    Task/proprioception and all other non-camera data are copied unchanged.
    """
    if type(raw) is not dict or any(camera not in raw for camera in CAMERAS):
        raise ValueError("Raw observation must contain both canonical RGB cameras")
    before = {camera: _pixels(raw[camera], camera) for camera in CAMERAS}
    operations = validate_image_perturbations(
        operations,
        {camera: pixels.shape for camera, pixels in before.items()},
        limits=limits,
    )
    result = copy.deepcopy(raw)
    for camera, pixels in before.items():
        result[camera] = pixels.copy()
    records = {}
    for operation in operations:
        camera = operation["camera"]
        pixels = before[camera]
        record = {"operation": copy.deepcopy(operation), "donor_provenance": None}
        height, width, _ = pixels.shape
        if operation["kind"] == "demo_blend":
            donor, provenance = _donor(
                donor_lookup, operation["donor_id"], camera, pixels.shape
            )
            result[camera] = _convex_uint8(pixels, donor, operation["alpha"])
            record["donor_provenance"] = provenance
            record["mask_pixels"] = height * width
        else:
            x0, y0, x1, y1 = operation["box_xyxy"]
            result[camera][y0:y1, x0:x1] = _convex_uint8(
                pixels[y0:y1, x0:x1], operation["fill_rgb"], operation["strength"]
            )
            record["mask_pixels"] = (x1 - x0) * (y1 - y0)
        record["mask_fraction"] = record["mask_pixels"] / (width * height)
        records[camera] = record
    unchanged = {key: value for key, value in raw.items() if key not in CAMERAS}
    non_image_digest = digest(unchanged)
    if (
        digest({key: value for key, value in result.items() if key not in CAMERAS})
        != non_image_digest
    ):
        raise AssertionError(
            "Image perturbation modified a non-camera observation field"
        )
    camera_audits = {}
    for camera in CAMERAS:
        metrics = _differences(before[camera], result[camera])
        camera_audits[camera] = {
            "shape": list(before[camera].shape),
            "dtype": "uint8",
            "before_sha256": digest(before[camera]),
            "after_sha256": digest(result[camera]),
            "has_effect": metrics["changed_pixels"] > 0,
            **metrics,
            **records.get(
                camera,
                {
                    "operation": None,
                    "donor_provenance": None,
                    "mask_pixels": 0,
                    "mask_fraction": 0.0,
                },
            ),
        }
    return result, {
        "schema_version": SCHEMA_VERSION,
        "rounding": ROUNDING,
        "limits": limits.as_dict(),
        "operations_sha256": digest(operations),
        "before_observation_sha256": digest(raw),
        "after_observation_sha256": digest(result),
        "non_image_sha256": non_image_digest,
        "has_effect": any(row["has_effect"] for row in camera_audits.values()),
        "cameras": camera_audits,
    }
