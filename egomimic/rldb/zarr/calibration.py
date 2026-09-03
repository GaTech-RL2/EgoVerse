"""Read and write the per-episode ``calibration`` metadata block.

Calibration is a physical measurement of the rig that recorded one episode, so
it travels with the episode instead of living in a class constant. An episode
stores it under one ``calibration`` attribute::

    "calibration": {
        "reference_frame": "camera:front_1",
        "cameras": {
            "front_1": {
                "K": [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]],
                "resolution": [W, H],
                "rectified": true,
                "ref_T_cam": [[...]],
            },
        },
        "arm_bases": {"left": [[...]], "right": [[...]]},
    }

Every matrix follows ``docs/CONVENTIONS.md``: ``A_T_B`` maps coordinates from
frame ``B`` to frame ``A``. ``ref_T_cam`` is the camera pose in the reference
frame, and ``arm_bases[side]`` is ``ref_T_armbase``, the arm base pose in the
reference frame.

Episodes written before this block existed store ``intrinsics`` and
``extrinsics`` at the top level instead. :func:`read_calibration` reads either
form, so no stored episode needs a rewrite.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Camera that legacy episodes calibrate and express `extrinsics` against.
LEGACY_REFERENCE_CAMERA = "front_1"

#: Reference frames that do not name a camera.
STATIC_REFERENCE_FRAMES = frozenset({"robot_base", "slam_world"})

#: Prefix that makes a camera the reference frame, as in `camera:front_1`.
CAMERA_FRAME_PREFIX = "camera:"

#: Prefix of the array key that stores one camera stream, as in
#: `images.front_1`. The text after it is the camera name.
IMAGE_KEY_PREFIX = "images."

#: Projection model of a camera, and the distortion coefficient counts it
#: accepts. Nothing projects a non-pinhole model yet. The declaration is
#: collected now because it measures a vendor's rig: if we discover later that
#: we need it, the rig has moved and the episodes cannot be recalibrated.
CAMERA_MODELS = {
    "PINHOLE": frozenset({0}),
    "OPENCV": frozenset({4, 5, 8, 12, 14}),
    "KANNALA_BRANDT": frozenset({4}),
}

#: Projection model assumed for a camera that declares none.
DEFAULT_CAMERA_MODEL = "PINHOLE"

_CAMERA_FIELDS = frozenset(
    {"K", "model", "distortion", "resolution", "rectified", "ref_T_cam"}
)
_CALIBRATION_FIELDS = frozenset({"reference_frame", "cameras", "arm_bases"})


class CalibrationError(ValueError):
    """Report an invalid ``calibration`` block or legacy calibration attribute."""


def _matrix(value: Any, shape: tuple[int, ...], where: str) -> np.ndarray:
    """Return ``value`` as a finite float64 array of the given shape."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{where}: not a numeric array ({exc})") from exc
    if arr.shape != shape:
        raise CalibrationError(
            f"{where}: expected shape {shape}, got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise CalibrationError(f"{where}: contains a non-finite value")
    return arr


def _camera_matrix(value: Any, where: str) -> np.ndarray:
    """Return a 3×4 camera matrix, padding a bare 3×3 with a zero column."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{where}: not a numeric array ({exc})") from exc
    if arr.shape == (3, 3):
        arr = np.hstack([arr, np.zeros((3, 1))])
    if arr.shape != (3, 4):
        raise CalibrationError(
            f"{where}: expected a 3x4 camera matrix (pad a bare 3x3 with "
            f"np.hstack([K, np.zeros((3, 1))])), got shape {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise CalibrationError(f"{where}: contains a non-finite value")
    return arr


@dataclass(frozen=True)
class CameraCalibration:
    """Store the calibration of one camera stream.

    Attributes:
        name: The camera name. It matches the ``images.<name>`` array key.
        K: The 3×4 camera matrix, or ``None`` for a declared but uncalibrated
            camera.
        model: A key in ``CAMERA_MODELS``. No projection site honors a
            non-pinhole model yet.
        distortion: The distortion coefficients of ``model``, in that model's
            order. Empty for a pinhole camera.
        resolution: ``(width, height)`` in pixels, or ``None``.
        rectified: Whether the stored frames are already rectified.
        ref_T_cam: The 4×4 camera pose in the episode reference frame, or
            ``None`` when the episode does not state it.
    """

    name: str
    K: np.ndarray | None = None
    model: str = DEFAULT_CAMERA_MODEL
    distortion: tuple[float, ...] = ()
    resolution: tuple[int, int] | None = None
    rectified: bool = True
    ref_T_cam: np.ndarray | None = None

    def to_jsonable(self) -> dict[str, Any]:
        """Return this camera as JSON-serializable episode metadata."""
        out: dict[str, Any] = {
            "model": self.model,
            "rectified": bool(self.rectified),
        }
        if self.K is not None:
            out["K"] = self.K.tolist()
        if self.distortion:
            out["distortion"] = [float(c) for c in self.distortion]
        if self.resolution is not None:
            out["resolution"] = [int(self.resolution[0]), int(self.resolution[1])]
        if self.ref_T_cam is not None:
            out["ref_T_cam"] = self.ref_T_cam.tolist()
        return out


@dataclass(frozen=True)
class Calibration:
    """Store the calibration of every camera and arm base in one episode.

    Attributes:
        reference_frame: ``robot_base``, ``slam_world``, or
            ``camera:<camera name>``. Every pose in this block is expressed in
            this frame.
        cameras: A mapping from camera name to its calibration.
        arm_bases: A mapping from ``"left"`` or ``"right"`` to a 4×4
            ``ref_T_armbase`` matrix. Human episodes leave this empty.
        legacy: Whether :func:`read_calibration` built this from the
            ``intrinsics`` and ``extrinsics`` attributes of an older episode.
    """

    reference_frame: str
    cameras: Mapping[str, CameraCalibration] = field(default_factory=dict)
    arm_bases: Mapping[str, np.ndarray] = field(default_factory=dict)
    legacy: bool = False

    @property
    def reference_camera(self) -> str | None:
        """Return the camera that defines the reference frame, if any."""
        if self.reference_frame.startswith(CAMERA_FRAME_PREFIX):
            return self.reference_frame[len(CAMERA_FRAME_PREFIX) :]
        return None

    def default_camera(self) -> str | None:
        """Return the camera to project against when a caller names none.

        The reference camera wins, then ``front_1``, then any camera whose name
        contains ``front``, then the first declared camera. Projection and
        visualization target the front image, so a front camera outranks a
        wrist camera that happens to be declared first.
        """
        reference = self.reference_camera
        if reference is not None and reference in self.cameras:
            return reference
        if LEGACY_REFERENCE_CAMERA in self.cameras:
            return LEGACY_REFERENCE_CAMERA
        front = next((n for n in self.cameras if "front" in n.lower()), None)
        if front is not None:
            return front
        return next(iter(self.cameras), None)

    def K(self, camera: str | None = None) -> np.ndarray | None:
        """Return the 3×4 camera matrix of one camera.

        Args:
            camera: A camera name. ``None`` selects :meth:`default_camera`.

        Returns:
            The camera matrix, or ``None`` if the camera is absent or carries
            no ``K``.
        """
        name = camera if camera is not None else self.default_camera()
        if name is None:
            return None
        entry = self.cameras.get(name)
        return None if entry is None else entry.K

    def ref_T_cam(self, camera: str | None = None) -> np.ndarray | None:
        """Return the pose of one camera in the reference frame.

        The reference camera is the identity by definition, so an episode that
        references a camera need not state that camera's pose.
        """
        name = camera if camera is not None else self.default_camera()
        if name is None:
            return None
        entry = self.cameras.get(name)
        if entry is not None and entry.ref_T_cam is not None:
            return entry.ref_T_cam
        if name == self.reference_camera:
            return np.eye(4)
        return None

    def base_T_cam(self, side: str, camera: str | None = None) -> np.ndarray | None:
        """Return one camera's pose in the frame of one arm base.

        This is the quantity the EVA transform pipeline consumes, and it is
        what the ``extrinsics`` attribute of a legacy episode stores directly.

        Args:
            side: ``"left"`` or ``"right"``.
            camera: A camera name. ``None`` selects :meth:`default_camera`.

        Returns:
            A 4×4 ``armbase_T_cam`` matrix, or ``None`` when the episode
            states no arm base or no camera pose.
        """
        ref_T_armbase = self.arm_bases.get(side)
        ref_T_cam = self.ref_T_cam(camera)
        if ref_T_armbase is None or ref_T_cam is None:
            return None
        return np.linalg.inv(ref_T_armbase) @ ref_T_cam

    def intrinsics(self) -> dict[str, np.ndarray]:
        """Return ``{camera: K}`` for every camera that carries a ``K``."""
        return {
            name: cam.K for name, cam in self.cameras.items() if cam.K is not None
        }

    def extrinsics(self, camera: str | None = None) -> dict[str, np.ndarray]:
        """Return ``{side: base_T_cam}`` in the legacy attribute layout."""
        out = {}
        for side in self.arm_bases:
            base_T_cam = self.base_T_cam(side, camera)
            if base_T_cam is not None:
                out[side] = base_T_cam
        return out

    def to_jsonable(self) -> dict[str, Any]:
        """Return this calibration as JSON-serializable episode metadata."""
        out: dict[str, Any] = {
            "reference_frame": self.reference_frame,
            "cameras": {
                name: cam.to_jsonable() for name, cam in self.cameras.items()
            },
        }
        if self.arm_bases:
            out["arm_bases"] = {
                side: np.asarray(T, dtype=np.float64).tolist()
                for side, T in self.arm_bases.items()
            }
        return out


def camera_name(image_key: str) -> str | None:
    """Return the camera that one stored image array belongs to.

    Args:
        image_key: An array key such as ``"images.front_1"``.

    Returns:
        The camera name, or ``None`` if the key names no image stream.
    """
    marker = IMAGE_KEY_PREFIX
    index = image_key.rfind(marker)
    if index == -1:
        return None
    return image_key[index + len(marker) :] or None


def uncalibrated_cameras(image_keys, calibration: Calibration | None) -> list[str]:
    """Return the image streams that no camera matrix covers.

    Coverage is the rule that makes per-episode calibration real: an episode
    that stores three image streams and one ``K`` calibrates one camera in
    three, and nothing downstream can tell.

    Args:
        image_keys: The episode's image array keys.
        calibration: The episode calibration, or ``None``.

    Returns:
        The camera names, in the order the keys give them and without
        duplicates, that carry no ``K``.
    """
    missing: list[str] = []
    seen: set[str] = set()
    for key in image_keys:
        name = camera_name(key)
        if name is None or name in seen:
            continue
        seen.add(name)
        if calibration is None or calibration.K(name) is None:
            missing.append(name)
    return missing


def _parse_reference_frame(value: Any, cameras, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise CalibrationError(f"{where}: `reference_frame` must be a non-empty string")
    if value in STATIC_REFERENCE_FRAMES:
        return value
    if value.startswith(CAMERA_FRAME_PREFIX):
        camera = value[len(CAMERA_FRAME_PREFIX) :]
        if camera and camera in cameras:
            return value
        raise CalibrationError(
            f"{where}: reference_frame {value!r} names a camera that this episode "
            f"does not declare; declared cameras are {sorted(cameras)}"
        )
    raise CalibrationError(
        f"{where}: unknown reference_frame {value!r}; expected one of "
        f"{sorted(STATIC_REFERENCE_FRAMES)} or 'camera:<declared camera>'"
    )


def _parse_distortion(raw: Any, model: str, where: str) -> tuple[float, ...]:
    """Validate the distortion coefficients declared for one camera model."""
    if raw is None:
        raw = ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple, np.ndarray)):
        raise CalibrationError(
            f"{where}.distortion: expected a list of coefficients, got {raw!r}"
        )
    try:
        coefficients = tuple(float(c) for c in raw)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(
            f"{where}.distortion: contains a non-numeric coefficient ({exc})"
        ) from exc
    if not all(np.isfinite(coefficients)):
        raise CalibrationError(f"{where}.distortion: contains a non-finite value")
    allowed = CAMERA_MODELS[model]
    if len(coefficients) not in allowed:
        raise CalibrationError(
            f"{where}.distortion: model {model} takes "
            f"{sorted(allowed)} coefficients, got {len(coefficients)}"
        )
    return coefficients


def _parse_camera(name: str, block: Any, where: str) -> CameraCalibration:
    if not isinstance(block, Mapping):
        raise CalibrationError(f"{where}: must be a mapping, got {block!r}")
    unknown = sorted(set(block) - _CAMERA_FIELDS)
    if unknown:
        raise CalibrationError(
            f"{where}: unknown field(s) {unknown}; expected some of "
            f"{sorted(_CAMERA_FIELDS)}"
        )

    K = block.get("K")

    model = block.get("model", DEFAULT_CAMERA_MODEL)
    if model not in CAMERA_MODELS:
        raise CalibrationError(
            f"{where}.model: unknown camera model {model!r}; expected one of "
            f"{sorted(CAMERA_MODELS)}"
        )
    distortion = _parse_distortion(block.get("distortion"), model, where)

    resolution = block.get("resolution")
    if resolution is not None:
        if (
            not isinstance(resolution, (list, tuple))
            or len(resolution) != 2
            or not all(isinstance(v, int) and v > 0 for v in resolution)
        ):
            raise CalibrationError(
                f"{where}.resolution: expected [width, height] positive ints, "
                f"got {resolution!r}"
            )
        resolution = (int(resolution[0]), int(resolution[1]))

    rectified = block.get("rectified", True)
    if not isinstance(rectified, bool):
        raise CalibrationError(
            f"{where}.rectified: expected a boolean, got {rectified!r}"
        )

    ref_T_cam = block.get("ref_T_cam")
    return CameraCalibration(
        name=name,
        K=None if K is None else _camera_matrix(K, f"{where}.K"),
        model=model,
        distortion=distortion,
        resolution=resolution,
        rectified=rectified,
        ref_T_cam=(
            None
            if ref_T_cam is None
            else _matrix(ref_T_cam, (4, 4), f"{where}.ref_T_cam")
        ),
    )


def parse_calibration(block: Any, where: str = "calibration") -> Calibration:
    """Validate one ``calibration`` attribute block.

    Args:
        block: The mapping stored under ``zarr.attrs["calibration"]``.
        where: A label used in error messages.

    Returns:
        The parsed calibration.

    Raises:
        CalibrationError: If a field is unknown, missing, or malformed.
    """
    if isinstance(block, Calibration):
        return block
    if not isinstance(block, Mapping):
        raise CalibrationError(f"{where}: must be a mapping, got {block!r}")
    unknown = sorted(set(block) - _CALIBRATION_FIELDS)
    if unknown:
        raise CalibrationError(
            f"{where}: unknown field(s) {unknown}; expected some of "
            f"{sorted(_CALIBRATION_FIELDS)}"
        )

    raw_cameras = block.get("cameras")
    if not isinstance(raw_cameras, Mapping) or not raw_cameras:
        raise CalibrationError(
            f"{where}.cameras: expected a non-empty mapping from camera name to "
            f"its calibration, got {raw_cameras!r}"
        )
    cameras = {
        name: _parse_camera(name, cam, f"{where}.cameras[{name!r}]")
        for name, cam in raw_cameras.items()
    }

    if "reference_frame" not in block:
        raise CalibrationError(f"{where}: missing required field 'reference_frame'")
    reference_frame = _parse_reference_frame(
        block["reference_frame"], cameras, where
    )

    raw_arm_bases = block.get("arm_bases") or {}
    if not isinstance(raw_arm_bases, Mapping):
        raise CalibrationError(
            f"{where}.arm_bases: expected a mapping from side to a 4x4 "
            f"ref_T_armbase matrix, got {raw_arm_bases!r}"
        )
    arm_bases = {
        side: _matrix(T, (4, 4), f"{where}.arm_bases[{side!r}]")
        for side, T in raw_arm_bases.items()
    }

    return Calibration(
        reference_frame=reference_frame, cameras=cameras, arm_bases=arm_bases
    )


def lift_legacy_calibration(
    intrinsics: Any = None, extrinsics: Any = None
) -> Calibration | None:
    """Build a :class:`Calibration` from the legacy attribute pair.

    ``intrinsics`` maps a camera name to its ``K``. A bare matrix is read as
    the ``front_1`` camera, which is the only camera any legacy writer
    calibrated. ``extrinsics`` maps an arm side to ``base_T_cam``, the front
    camera's pose in that arm's base frame, so the reference frame is the front
    camera and ``arm_bases[side]`` is the inverse of the stored matrix.

    Args:
        intrinsics: The ``intrinsics`` attribute, or ``None``.
        extrinsics: The ``extrinsics`` attribute, or ``None``.

    Returns:
        The lifted calibration, or ``None`` if both attributes are absent.

    Raises:
        CalibrationError: If either attribute is malformed.
    """
    if not intrinsics and not extrinsics:
        return None

    if isinstance(intrinsics, Mapping):
        raw_cameras = list(intrinsics.items())
    elif intrinsics is not None:
        raw_cameras = [(LEGACY_REFERENCE_CAMERA, intrinsics)]
    else:
        raw_cameras = []

    cameras = {
        str(name): CameraCalibration(
            name=str(name), K=_camera_matrix(K, f"intrinsics[{name!r}]")
        )
        for name, K in raw_cameras
    }

    arm_bases: dict[str, np.ndarray] = {}
    if extrinsics is not None:
        if not isinstance(extrinsics, Mapping):
            raise CalibrationError(
                f"extrinsics: expected a mapping from arm side to a 4x4 "
                f"base_T_cam matrix, got {extrinsics!r}"
            )
        for side, base_T_cam in extrinsics.items():
            matrix = _matrix(base_T_cam, (4, 4), f"extrinsics[{side!r}]")
            arm_bases[str(side)] = np.linalg.inv(matrix)

    # The legacy `extrinsics` values are per-arm poses of the front camera, so
    # that camera is the reference frame. Declare it even when the episode
    # calibrated no camera, otherwise `base_T_cam` has no pose to compose.
    if arm_bases and LEGACY_REFERENCE_CAMERA not in cameras:
        cameras[LEGACY_REFERENCE_CAMERA] = CameraCalibration(
            name=LEGACY_REFERENCE_CAMERA
        )

    reference_frame = (
        f"{CAMERA_FRAME_PREFIX}{LEGACY_REFERENCE_CAMERA}"
        if LEGACY_REFERENCE_CAMERA in cameras
        else f"{CAMERA_FRAME_PREFIX}{next(iter(cameras))}"
    )
    return Calibration(
        reference_frame=reference_frame,
        cameras=cameras,
        arm_bases=arm_bases,
        legacy=True,
    )


def read_calibration(attrs: Mapping[str, Any]) -> Calibration | None:
    """Return the calibration of one episode from its Zarr attributes.

    This is the one reader every consumer goes through. It prefers the
    ``calibration`` block and falls back to the legacy ``intrinsics`` and
    ``extrinsics`` attributes, so episodes written before the block existed
    stay readable without a rewrite.

    Args:
        attrs: The episode's Zarr attributes.

    Returns:
        The episode calibration, or ``None`` if the episode states none.

    Raises:
        CalibrationError: If the stored calibration is malformed.
    """
    block = attrs.get("calibration")
    if block:
        return parse_calibration(block)
    return lift_legacy_calibration(
        attrs.get("intrinsics"), attrs.get("extrinsics")
    )


__all__ = [
    "CAMERA_FRAME_PREFIX",
    "CAMERA_MODELS",
    "DEFAULT_CAMERA_MODEL",
    "IMAGE_KEY_PREFIX",
    "LEGACY_REFERENCE_CAMERA",
    "STATIC_REFERENCE_FRAMES",
    "Calibration",
    "CalibrationError",
    "CameraCalibration",
    "camera_name",
    "lift_legacy_calibration",
    "parse_calibration",
    "read_calibration",
    "uncalibrated_cameras",
]
