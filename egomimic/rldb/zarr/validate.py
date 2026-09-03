"""Validate one zarr episode against the schema in ``schema/episode_v3.yaml``.

Run it with::

    python -m egomimic.rldb.zarr.validate <episode.zarr> [--strict]

The rules live in the schema file, not here. This module reads them, resolves
the episode's embodiment through the registry, and reports one finding per
rule. ``--strict`` promotes the rules the corpus does not meet yet from
warnings to errors, so a rule can land, be measured across the corpus, and
only then become the default.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import zarr

from egomimic.rldb.embodiment.embodiment import (
    Embodiment,
    ResolvedEmbodiment,
    canonical_embodiment_name,
)
from egomimic.rldb.embodiment.registry import load_embodiment_platforms
from egomimic.rldb.zarr.calibration import (
    IMAGE_KEY_PREFIX,
    CalibrationError,
    read_calibration,
    uncalibrated_cameras,
)

SCHEMA_DIR = Path(__file__).parent / "schema"
SCHEMA_FILE = SCHEMA_DIR / "episode_v3.yaml"

ERROR = "error"
WARNING = "warning"
OK = "ok"

#: `required: strict` rules are warnings until `--strict` promotes them. They
#: exist so a rule can land before the corpus meets it.
_REQUIRED_VALUES = (True, False, "strict")

_TYPE_NAMES = {
    "str": str,
    "int": int,
    "bool": bool,
    "mapping": Mapping,
}


class SchemaError(ValueError):
    """Report an invalid rule in ``episode_v3.yaml``."""


@dataclass(frozen=True)
class Finding:
    """One rule's result.

    Attributes:
        level: ``ok``, ``warning``, or ``error``.
        check: The rule that produced this finding.
        message: What the rule found, in one line.
    """

    level: str
    check: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():<7} {self.check}: {self.message}"


@dataclass
class Report:
    """Every finding for one episode.

    Attributes:
        path: The episode directory.
        findings: One finding per rule that ran, in schema order.
        strict: Whether the strict rules were promoted to errors.
    """

    path: Path
    findings: list[Finding] = field(default_factory=list)
    strict: bool = False

    def add(self, level: str, check: str, message: str) -> None:
        self.findings.append(Finding(level, check, message))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return (
            f"{len(self.findings)} checks, {len(self.errors)} errors, "
            f"{len(self.warnings)} warnings"
        )

    def text(self, verbose: bool = False) -> str:
        """Render the report.

        Args:
            verbose: If true, list the rules that passed as well.
        """
        lines = [str(self.path)]
        for finding in self.findings:
            if finding.level != OK or verbose:
                lines.append(f"  {finding}")
        lines.append(f"  {self.summary()}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "strict": self.strict,
            "ok": self.ok,
            "findings": [
                {"level": f.level, "check": f.check, "message": f.message}
                for f in self.findings
            ],
        }


@functools.lru_cache(maxsize=1)
def load_schema() -> dict:
    """Load and check ``schema/episode_v3.yaml``.

    Returns:
        The parsed schema.

    Raises:
        SchemaError: If a rule declares an unusable ``required`` value.
    """
    with SCHEMA_FILE.open("r") as f:
        schema = yaml.safe_load(f) or {}
    rules = list(schema.get("attributes", {}).items())
    rules += [(r.get("name"), r) for r in schema.get("checks", [])]
    rules += [(r.get("key"), r) for r in schema.get("arrays", [])]
    for name, rule in rules:
        required = rule.get("required", False)
        if required not in _REQUIRED_VALUES:
            raise SchemaError(
                f"{name!r}: `required` must be one of "
                f"{list(_REQUIRED_VALUES)}, got {required!r}"
            )
    return schema


def _level(required, strict: bool) -> str | None:
    """Return the level a failed rule reports at, or ``None`` if it is optional."""
    if required is True:
        return ERROR
    if required == "strict":
        return ERROR if strict else WARNING
    return None


# ---------------------------------------------------------------------------
# Attribute rules
# ---------------------------------------------------------------------------


def _check_type(value, type_name: str | None) -> str | None:
    if type_name is None:
        return None
    expected = _TYPE_NAMES.get(type_name)
    if expected is None:
        raise SchemaError(f"unknown type {type_name!r} in the schema")
    # A bool is an int in Python; the schema means them separately.
    if expected is int and isinstance(value, bool):
        return "expected an int, got a bool"
    if not isinstance(value, expected):
        return f"expected {type_name}, got {type(value).__name__}"
    return None


def _check_attribute(name: str, rule: dict, attrs: Mapping, report, context) -> None:
    if name not in attrs:
        level = _level(rule.get("required", False), report.strict)
        if level is not None:
            report.add(level, f"attrs.{name}", "missing")
        return

    value = attrs[name]
    problem = _check_type(value, rule.get("type"))
    if problem is None and "min" in rule and value < rule["min"]:
        problem = f"must be at least {rule['min']}, got {value}"
    if problem is None and "choices" in rule and value not in rule["choices"]:
        problem = f"expected one of {rule['choices']}, got {value!r}"
    if problem is None and rule.get("check"):
        problem = _ATTRIBUTE_CHECKS[rule["check"]](value, attrs, context)

    if problem is None:
        report.add(OK, f"attrs.{name}", _describe(value))
    else:
        report.add(ERROR, f"attrs.{name}", problem)


def _describe(value) -> str:
    if isinstance(value, Mapping):
        return f"{len(value)} entries"
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _check_embodiment_name(value, attrs, context) -> str | None:
    if canonical_embodiment_name(value) not in load_embodiment_platforms():
        return (
            f"{value!r} is not owned by any platform in registry/platforms.yaml; "
            f"known: {sorted(load_embodiment_platforms())}"
        )
    return None


def _check_morphology(value, attrs, context) -> str | None:
    try:
        resolved = Embodiment.resolve(value)
    except (TypeError, ValueError) as exc:
        return str(exc)
    named = context.get("named_platform")
    if named is not None and resolved.platform.name != named.name:
        return (
            f"platform {resolved.platform.name!r} disagrees with the platform "
            f"{named.name!r} that embodiment {attrs.get('embodiment')!r} selects"
        )
    return None


def _check_calibration(value, attrs, context) -> str | None:
    try:
        read_calibration(attrs)
    except CalibrationError as exc:
        return str(exc)
    return None


def _check_schema_version(value, attrs, context) -> str | None:
    known = load_schema().get("known_versions") or []
    if known and value not in known:
        return f"unknown schema_version {value!r}; known: {known}"
    return None


_ATTRIBUTE_CHECKS = {
    "embodiment_name": _check_embodiment_name,
    "morphology": _check_morphology,
    "calibration": _check_calibration,
    "schema_version": _check_schema_version,
}


# ---------------------------------------------------------------------------
# Named checks
# ---------------------------------------------------------------------------


def _check_calibration_present(rule, context, report) -> None:
    calibration = context.get("calibration")
    level = _level(rule.get("required", False), report.strict)
    if calibration is None or not calibration.intrinsics():
        if level is not None:
            report.add(
                level,
                "calibration_present",
                "the episode states no camera matrix; calibration measures the "
                "rig that recorded it and cannot be recovered later",
            )
        return
    report.add(
        OK,
        "calibration_present",
        f"{len(calibration.intrinsics())} calibrated camera(s), "
        f"reference_frame={calibration.reference_frame}",
    )


def _check_camera_coverage(rule, context, report) -> None:
    missing = uncalibrated_cameras(context["array_keys"], context.get("calibration"))
    level = _level(rule.get("required", False), report.strict)
    if missing:
        if level is not None:
            report.add(
                level,
                "camera_coverage",
                f"no camera matrix for image stream(s) {missing}",
            )
        return
    report.add(OK, "camera_coverage", "every image stream has a camera matrix")


_IDENTITY = np.eye(4)

#: Quaternion layout is `[qw, qx, qy, qz]`; both signs are the same rotation.
_IDENTITY_QUATERNIONS = (
    np.array([1.0, 0.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0, 0.0]),
)


def _read(array, total_frames: int | None) -> np.ndarray:
    """Read an array up to `total_frames`, which is the authoritative length."""
    end = array.shape[0] if total_frames is None else min(total_frames, array.shape[0])
    return np.asarray(array[:end])


def _report_problems(rule, report, check: str, problems, passed: str) -> None:
    """Record one finding for a named check."""
    if not problems:
        report.add(OK, check, passed)
        return
    level = _level(rule.get("required", False), report.strict)
    if level is not None:
        report.add(level, check, "; ".join(problems))


def _check_pose_degeneracy(rule, context, report) -> None:
    suffixes = tuple(rule.get("suffixes") or ())
    threshold = float(rule.get("identity_rotation_fraction", 0.01))
    total_frames = context.get("total_frames")
    problems = []
    checked = 0
    for key, array in context["arrays"].items():
        if not key.endswith(suffixes) or len(array.shape) != 2:
            continue
        poses = _read(array, total_frames)
        if poses.shape[0] < 2 or poses.shape[1] != 7:
            continue
        checked += 1
        if len(np.unique(poses, axis=0)) == 1:
            problems.append(f"{key} is constant across all {poses.shape[0]} frames")
        rotations = poses[:, 3:7]
        identity = np.zeros(len(rotations), dtype=bool)
        for quaternion in _IDENTITY_QUATERNIONS:
            identity |= np.all(rotations == quaternion, axis=1)
        fraction = float(identity.mean())
        if fraction > threshold:
            problems.append(
                f"{key} holds an exact identity rotation on "
                f"{fraction:.0%} of frames (limit {threshold:.0%})"
            )
    _report_problems(
        rule, report, "pose_degeneracy", problems, f"{checked} pose track(s) move"
    )


def _check_calibration_degeneracy(rule, context, report) -> None:
    calibration = context.get("calibration")
    if calibration is None:
        return
    problems = []
    for name, camera in calibration.cameras.items():
        # The reference camera is the identity by definition, so only a stored
        # pose can be degenerate.
        if (
            camera.ref_T_cam is not None
            and name != calibration.reference_camera
            and np.array_equal(camera.ref_T_cam, _IDENTITY)
        ):
            problems.append(f"cameras[{name!r}].ref_T_cam is exactly the identity")
    for side in calibration.arm_bases:
        base_T_cam = calibration.base_T_cam(side)
        if base_T_cam is not None and np.allclose(base_T_cam, _IDENTITY, atol=0.0):
            problems.append(
                f"the camera sits exactly at the {side} arm base "
                "(base_T_cam is the identity)"
            )
    _report_problems(
        rule, report, "calibration_degeneracy", problems, "no identity extrinsic"
    )


def _resolution(camera, context) -> tuple[int, int] | None:
    """Return one camera's ``(width, height)``, from the block or the stream."""
    if camera.resolution is not None:
        return camera.resolution
    feature = context["features"].get(f"{IMAGE_KEY_PREFIX}{camera.name}") or {}
    shape = feature.get("shape") or []
    if len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    return None


def _check_intrinsics_signature(rule, context, report) -> None:
    calibration = context.get("calibration")
    if calibration is None:
        return
    problems = []
    checked = 0
    for name, camera in calibration.cameras.items():
        resolution = _resolution(camera, context)
        if camera.K is None or resolution is None:
            continue
        checked += 1
        width, height = resolution
        fx, fy = camera.K[0, 0], camera.K[1, 1]
        cx, cy = camera.K[0, 2], camera.K[1, 2]
        if fx == fy == width and cx == width / 2 and cy == height / 2:
            problems.append(
                f"cameras[{name!r}].K is a synthesized centred pinhole "
                f"(fx = fy = {fx:g} = W, principal point at the image centre)"
            )
    _report_problems(
        rule,
        report,
        "intrinsics_signature",
        problems,
        f"{checked} camera matri(ces) carry a measured focal length",
    )


def _check_timestamps(rule, context, report) -> None:
    problems = []
    for substring in rule.get("banned_key_substrings") or ():
        stored = [k for k in context["arrays"] if substring in k]
        if stored:
            problems.append(
                f"{stored} store a second time base; keep one clock per episode "
                "and derive a relative base on read"
            )

    key = rule.get("key", "obs_rgb_timestamps_ns")
    array = context["arrays"].get(key)
    if array is None:
        _report_problems(rule, report, "timestamps", problems, "no clock stored")
        return

    stamps = _read(array, context.get("total_frames"))
    if stamps.ndim != 1 or stamps.shape[0] < 2:
        _report_problems(rule, report, "timestamps", problems, "one stamp")
        return

    steps = np.diff(stamps.astype(np.int64))
    stalled = int(np.count_nonzero(steps <= 0))
    if stalled:
        problems.append(f"{key} does not increase on {stalled} of {len(steps)} steps")

    quantum = int(np.gcd.reduce(np.abs(steps))) if steps.size else 0
    if quantum >= 64 and quantum & (quantum - 1) == 0:
        problems.append(
            f"{key} is quantized to {quantum} ns, the signature of float64 "
            "seconds converted to nanoseconds; compute the clock in integers"
        )

    _report_problems(
        rule,
        report,
        "timestamps",
        problems,
        f"{len(stamps)} stamps increase strictly",
    )


def _annotations(context, key: str) -> list[dict]:
    array = context["arrays"].get(key)
    if array is None:
        return []
    out = []
    for entry in np.asarray(array[:]):
        if isinstance(entry, (bytes, bytearray, memoryview)):
            entry = bytes(entry).decode("utf-8")
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except json.JSONDecodeError:
                continue
        if isinstance(entry, Mapping):
            out.append(dict(entry))
    return out


def _check_annotation_coverage(rule, context, report) -> None:
    key = rule.get("key", "annotations")
    total_frames = context.get("total_frames") or 0
    annotations = _annotations(context, key)
    problems = []
    covered = np.zeros(max(total_frames, 0), dtype=bool)
    for annotation in annotations:
        start = int(annotation.get("start_idx", -1))
        end = int(annotation.get("end_idx", -1))
        if start < 0 or end > total_frames or end <= start:
            problems.append(
                f"span [{start}, {end}) is outside [0, {total_frames}) or empty"
            )
            continue
        covered[start:end] = True
    fraction = float(covered.mean()) if covered.size else 0.0
    minimum = float(rule.get("minimum", 0.9))
    if fraction < minimum:
        problems.append(
            f"{len(annotations)} annotation(s) cover {fraction:.0%} of the "
            f"episode (minimum {minimum:.0%}); trim the tail or annotate it"
        )
    _report_problems(
        rule,
        report,
        "annotation_coverage",
        problems,
        f"{len(annotations)} annotation(s) cover {fraction:.0%}",
    )


def _check_annotation_text(rule, context, report) -> None:
    delimiters = tuple(rule.get("banned_delimiters") or ())
    texts = [
        (f"annotation {i}", a.get("text", ""))
        for i, a in enumerate(_annotations(context, rule.get("key", "annotations")))
    ]
    texts.append(("task_description", context["attrs"].get("task_description") or ""))
    problems = []
    for where, text in texts:
        for delimiter in delimiters:
            if delimiter in str(text):
                problems.append(
                    f"{where} encodes metadata after {delimiter!r}: {text!r}"
                )
                break
    _report_problems(
        rule, report, "annotation_text", problems, "no delimiter-encoded metadata"
    )


_NAMED_CHECKS = {
    "calibration_present": _check_calibration_present,
    "camera_coverage": _check_camera_coverage,
    "pose_degeneracy": _check_pose_degeneracy,
    "calibration_degeneracy": _check_calibration_degeneracy,
    "intrinsics_signature": _check_intrinsics_signature,
    "timestamps": _check_timestamps,
    "annotation_coverage": _check_annotation_coverage,
    "annotation_text": _check_annotation_text,
}


# ---------------------------------------------------------------------------
# Array rules
# ---------------------------------------------------------------------------


def _condition_holds(when: Mapping, resolved: ResolvedEmbodiment, side) -> bool:
    for name, expected in when.items():
        if name == "platform_kind":
            actual = resolved.platform.kind
        elif name == "has_arm_chain":
            actual = resolved.platform.arm_dof is not None
        elif name == "has_aux_chain":
            actual = resolved.platform.aux is not None
        elif name == "end_effector_class":
            end_effector = resolved.end_effectors.get(side) if side else None
            actual = None if end_effector is None else end_effector.ee_class
        else:
            raise SchemaError(f"unknown `when` condition {name!r} in the schema")
        if actual != expected:
            return False
    return True


def _dimension(token, context, side) -> int | None:
    """Resolve one schema shape token to a length, or ``None`` for unknown."""
    if isinstance(token, int):
        return token
    if token == "*":
        return None
    resolved: ResolvedEmbodiment = context["resolved"]
    if token == "T":
        return context["total_frames"]
    if token == "arm_dof":
        return resolved.platform.arm_dof
    if token == "aux_dof":
        return None if resolved.platform.aux is None else resolved.platform.aux.dof
    if token == "ee_dof":
        end_effector = resolved.end_effectors.get(side) if side else None
        return None if end_effector is None else end_effector.dof
    if token == "kp3":
        end_effector = resolved.end_effectors.get(side) if side else None
        if end_effector is None:
            return None
        return 3 * end_effector.keypoints.n_slots
    raise SchemaError(f"unknown shape dimension {token!r} in the schema")


def _dtype_kind(dtype) -> str:
    kind = getattr(dtype, "kind", "")
    if kind in "iu":
        return "int"
    if kind == "f":
        return "float"
    if kind in "OSV":
        return "object"
    return kind or "any"


def _check_array(key: str, rule: dict, arrays: Mapping, report, context, side) -> None:
    if key not in arrays:
        level = _level(rule.get("required", False), report.strict)
        if level is not None:
            report.add(level, key, "missing")
        return

    array = arrays[key]
    shape = tuple(int(n) for n in array.shape)
    expected = rule.get("shape")
    problems = []
    if expected is not None:
        if len(shape) != len(expected):
            problems.append(
                f"expected {len(expected)} dimension(s), got shape {shape}"
            )
        else:
            for axis, token in enumerate(expected):
                want = _dimension(token, context, side)
                if want is None:
                    continue
                # `total_frames` is the sole authoritative length and a stored
                # array may carry a padded tail, so axis 0 is a lower bound.
                if axis == 0:
                    if shape[0] < want:
                        problems.append(
                            f"holds {shape[0]} frames for total_frames {want}"
                        )
                elif shape[axis] != want:
                    problems.append(
                        f"axis {axis} is {shape[axis]}, expected {want} ({token})"
                    )

    want_dtype = rule.get("dtype")
    if want_dtype and want_dtype != "any":
        actual = _dtype_kind(array.dtype)
        if actual != want_dtype:
            problems.append(f"dtype is {array.dtype}, expected {want_dtype}")

    if problems:
        report.add(ERROR, key, "; ".join(problems))
    else:
        report.add(OK, key, f"shape {shape} {array.dtype}")


def _expand_key(template: str, arrays: Mapping, sides) -> list[tuple[str, str | None]]:
    """Expand one schema key into the concrete keys it names.

    ``{side}`` expands over the episode's arms. ``*`` matches the keys the
    episode stores, so a wildcard rule checks what is there and never demands
    a key.
    """
    if "{side}" in template:
        candidates = [(template.format(side=side), side) for side in sides]
    else:
        candidates = [(template, None)]
    out = []
    for key, side in candidates:
        if "*" not in key:
            out.append((key, side))
            continue
        prefix, _, suffix = key.partition("*")
        out.extend(
            (name, side)
            for name in arrays
            if name.startswith(prefix) and name.endswith(suffix)
        )
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def validate_episode(path: str | Path, *, strict: bool = False) -> Report:
    """Validate one zarr episode against ``schema/episode_v3.yaml``.

    Args:
        path: The episode ``.zarr`` directory.
        strict: If true, promote the rules the corpus does not meet yet from
            warnings to errors.

    Returns:
        A report holding one finding per rule that ran.
    """
    path = Path(path)
    report = Report(path=path, strict=strict)
    try:
        store = zarr.open_group(str(path), mode="r")
    except Exception as exc:
        report.add(ERROR, "episode", f"cannot open as a zarr group: {exc}")
        return report

    schema = load_schema()
    attrs = dict(store.attrs)
    arrays = {name: store[name] for name in store.array_keys()}

    context: dict[str, Any] = {
        "array_keys": list(arrays),
        "arrays": arrays,
        "attrs": attrs,
        "features": attrs.get("features") or {},
        "total_frames": attrs.get("total_frames"),
    }
    embodiment = attrs.get("embodiment")
    if embodiment:
        context["named_platform"] = load_embodiment_platforms().get(
            canonical_embodiment_name(embodiment)
        )
    try:
        context["calibration"] = read_calibration(attrs)
    except CalibrationError:
        context["calibration"] = None

    for name, rule in schema.get("attributes", {}).items():
        _check_attribute(name, rule, attrs, report, context)

    for rule in schema.get("checks", []):
        check = _NAMED_CHECKS.get(rule.get("name"))
        if check is None:
            raise SchemaError(f"unknown check {rule.get('name')!r} in the schema")
        check(rule, context, report)

    resolved = _resolve(attrs, report)
    if resolved is None:
        return report
    # Say which platform and end-effectors the array rules ran against, so a
    # report explains itself without a second lookup.
    report.add(OK, "embodiment", resolved.describe())
    context["resolved"] = resolved

    for rule in schema.get("arrays", []):
        for key, side in _expand_key(rule["key"], arrays, resolved.sides):
            if side is not None and side not in resolved.end_effectors:
                continue
            when = rule.get("when") or {}
            if when and not _condition_holds(when, resolved, side):
                continue
            _check_array(key, rule, arrays, report, context, side)

    return report


def _resolve(attrs: Mapping, report: Report) -> ResolvedEmbodiment | None:
    """Resolve the episode's embodiment, preferring its morphology block."""
    for spec in (attrs.get("morphology"), attrs.get("embodiment")):
        if not spec:
            continue
        try:
            return Embodiment.resolve(spec)
        except (TypeError, ValueError):
            continue
    report.add(
        ERROR,
        "embodiment",
        "cannot resolve the episode's embodiment, so no array rule can run",
    )
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the validator over one or more episodes.

    Returns:
        ``0`` if every episode passed, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="python -m egomimic.rldb.zarr.validate",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("paths", nargs="+", type=Path, help="episode .zarr paths")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="promote the rules the corpus does not meet yet to errors",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="list the rules that passed as well",
    )
    parser.add_argument(
        "--json", action="store_true", help="print one JSON report per episode"
    )
    args = parser.parse_args(argv)

    reports = [validate_episode(p, strict=args.strict) for p in args.paths]
    if args.json:
        print(json.dumps([r.to_jsonable() for r in reports], indent=2))
    else:
        for report in reports:
            print(report.text(verbose=args.verbose))
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
