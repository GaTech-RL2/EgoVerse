"""Validate a Zarr episode against ``schema/episode_v3.yaml``.

Run it with::

    python -m egomimic.rldb.zarr.validate <episode.zarr> [RULE FLAGS]

The schema declares attributes, arrays, conditions, thresholds, and severity
classes. Integrity failures are always errors. Adoption and coverage failures
are warnings by default, and each such rule supplies a matched ``--<rule>`` /
``--no-<rule>`` CLI pair. Validation of a present attribute or array is always
an error when its declared type or shape is wrong.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
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
from egomimic.rldb.embodiment.hand_kinematics import keypoint_residuals
from egomimic.rldb.embodiment.registry import load_embodiment_platforms
from egomimic.rldb.embodiment.urdf import UrdfError
from egomimic.rldb.zarr.calibration import (
    IMAGE_KEY_PREFIX,
    CalibrationError,
    read_calibration,
    uncalibrated_cameras,
)
from egomimic.rldb.zarr.episode_attrs import data_status, is_complete

SCHEMA_DIR = Path(__file__).parent / "schema"
SCHEMA_FILE = SCHEMA_DIR / "episode_v3.yaml"

ERROR = "error"
WARNING = "warning"
OK = "ok"

#: Accepted values for a schema rule's ``severity`` class.
_SEVERITY_VALUES = ("integrity", "adoption", "coverage", "diagnostic")
_INTEGRITY = "integrity"
_RULE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

_TYPE_NAMES = {
    "str": str,
    "int": int,
    "bool": bool,
    "mapping": Mapping,
}


class SchemaError(ValueError):
    """Raised when ``episode_v3.yaml`` contains an unsupported declaration."""


@dataclass(frozen=True)
class Finding:
    """One validation result emitted for an episode.

    Attributes:
        level: ``ok``, ``warning``, or ``error``.
        check: The attribute, array key, or named predicate being reported.
        message: What the rule found, in one line.
    """

    level: str
    check: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():<7} {self.check}: {self.message}"


@dataclass
class Report:
    """Validation findings for one episode.

    Attributes:
        path: The episode directory.
        findings: Findings in validation order. A schema rule may emit zero,
            one, or multiple findings after key expansion.
        requirements: Whether each waivable rule is required. A false value
            means that the invocation reports that rule's failures as warnings.
    """

    path: Path
    findings: list[Finding] = field(default_factory=list)
    requirements: dict[str, bool] = field(default_factory=dict)
    data_status: str | None = None
    status_eligible: bool = False
    capabilities: dict[str, Any] = field(default_factory=dict)

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
            f"; data_status={self.data_status or 'unknown'}, "
            f"status eligible={'yes' if self.status_eligible else 'no'}"
        )

    def text(self, verbose: bool = False) -> str:
        """Format this report for terminal output.

        Args:
            verbose: If true, list the rules that passed as well.
        """
        lines = [str(self.path)]
        for finding in self.findings:
            if finding.level != OK or verbose:
                lines.append(f"  {finding}")
        for name, capability in self.capabilities.items():
            if capability["requested"]:
                result = {True: "available", False: "unavailable", None: "not checked"}[capability["available"]]
                lines.append(f"  requested {name}: {result}")
        lines.append(f"  {self.summary()}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "requirements": self.requirements,
            "ok": self.ok,
            "data_status": self.data_status,
            "status_eligible": self.status_eligible,
            "capabilities": self.capabilities,
            "findings": [
                {"level": f.level, "check": f.check, "message": f.message}
                for f in self.findings
            ],
        }


@functools.lru_cache(maxsize=1)
def load_schema() -> dict:
    """Load the schema and validate its requirement declarations.

    Returns:
        The parsed schema.

    Raises:
        SchemaError: If a rule has an invalid requirement or severity class.
    """
    with SCHEMA_FILE.open("r") as f:
        schema = yaml.safe_load(f) or {}
    waivable_names = set()
    for default_name, rule in _schema_rules(schema):
        required = rule.get("required", False)
        if not isinstance(required, bool):
            raise SchemaError(
                f"{default_name!r}: `required` must be true or false, got {required!r}"
            )
        severity = rule.get("severity")
        if severity not in _SEVERITY_VALUES:
            raise SchemaError(
                f"{default_name!r}: `severity` must be one of "
                f"{list(_SEVERITY_VALUES)}, got {severity!r}"
            )
        if severity in (_INTEGRITY, "diagnostic"):
            continue
        if not required:
            raise SchemaError(
                f"{default_name!r}: a waivable rule must be required by default"
            )
        name = _rule_name(default_name, rule)
        if not _RULE_NAME.fullmatch(name):
            raise SchemaError(
                f"{default_name!r}: rule name {name!r} must use snake_case"
            )
        if name in waivable_names:
            raise SchemaError(f"duplicate waivable rule name {name!r}")
        waivable_names.add(name)
        if not isinstance(rule.get("why"), str) or not rule["why"].strip():
            raise SchemaError(
                f"{default_name!r}: a waivable rule needs a non-empty `why`"
            )
    return schema


def _schema_rules(schema: Mapping):
    """Yield ``(default_name, rule)`` pairs in schema order."""
    yield from schema.get("attributes", {}).items()
    yield from ((rule.get("name"), rule) for rule in schema.get("checks", []))
    yield from ((rule.get("key"), rule) for rule in schema.get("arrays", []))


def _rule_name(default_name: str, rule: Mapping) -> str:
    """Return the stable name used by requirement overrides and CLI flags."""
    return rule.get("name", default_name)


def waivable_rules(schema: Mapping | None = None) -> dict[str, dict]:
    """Return the schema rules whose failure severity callers may lower.

    The schema owns the rule names and explanations so the validator cannot add
    a waivable rule without also exposing a documented CLI flag for it.
    """
    schema = load_schema() if schema is None else schema
    return {
        _rule_name(default_name, rule): rule
        for default_name, rule in _schema_rules(schema)
        if rule.get("severity") in ("adoption", "coverage")
    }


def _requirements(
    schema: Mapping, overrides: Mapping[str, bool] | None
) -> dict[str, bool]:
    """Build one explicit error-or-warning decision per waivable rule."""
    requirements = dict.fromkeys(waivable_rules(schema), False)
    for name, required in (overrides or {}).items():
        if name not in requirements:
            raise SchemaError(
                f"unknown waivable rule {name!r}; expected one of "
                f"{sorted(requirements)}"
            )
        if not isinstance(required, bool):
            raise SchemaError(
                f"requirement for {name!r} must be a bool, got {required!r}"
            )
        requirements[name] = required
    return requirements


def _level(rule: Mapping, report: Report, default_name: str) -> str | None:
    """Return a missing or failed requirement's invocation-specific level."""
    if not rule.get("required", False):
        return None
    if rule["severity"] == _INTEGRITY:
        return ERROR
    if rule["severity"] == "diagnostic":
        return WARNING
    name = _rule_name(default_name, rule)
    return ERROR if report.requirements[name] else WARNING


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
        level = _level(rule, report, name)
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
        resolved = Embodiment.from_attrs(attrs)
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
    level = _level(rule, report, "calibration_present")
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
    level = _level(rule, report, "camera_coverage")
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
    """Read the available prefix, capped at the authoritative frame count."""
    end = array.shape[0] if total_frames is None else min(total_frames, array.shape[0])
    return np.asarray(array[:end])


def _report_problems(rule, report, check: str, problems, passed: str) -> None:
    """Record a named check's pass or its requirement-level failure."""
    if not problems:
        report.add(OK, check, passed)
        return
    level = _level(rule, report, check)
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
        # Identity is implicit for the reference camera; only an explicitly
        # stored pose for another camera is subject to this branch.
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
    """Return ``(width, height)`` from calibration or image feature metadata."""
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
                f"cameras[{name!r}].K matches a synthesized centred pinhole signature "
                f"(fx = fy = {fx:g} = W, principal point at the image centre)"
            )
    _report_problems(
        rule,
        report,
        "intrinsics_signature",
        problems,
        f"{checked} camera matri(ces) checked for synthetic signatures",
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
        _report_problems(
            {**rule, "severity": "diagnostic"},
            report,
            "timestamp_diagnostics",
            problems,
            "no clock stored",
        )
        return

    stamps = _read(array, context.get("total_frames"))
    if stamps.ndim != 1 or stamps.shape[0] < 2:
        _report_problems(
            {**rule, "severity": "diagnostic"},
            report,
            "timestamp_diagnostics",
            problems,
            "fewer than two stamps",
        )
        return

    steps = np.diff(stamps.astype(np.int64))
    stalled = int(np.count_nonzero(steps <= 0))
    if stalled:
        report.add(
            ERROR,
            "timestamps",
            f"{key} does not increase on {stalled} of {len(steps)} steps",
        )

    quantum = int(np.gcd.reduce(np.abs(steps))) if steps.size else 0
    if quantum >= 64 and quantum & (quantum - 1) == 0:
        problems.append(
            f"{key} is quantized to {quantum} ns, the signature of float64 "
            "seconds converted to nanoseconds; compute the clock in integers"
        )

    _report_problems(
        {**rule, "severity": "diagnostic"},
        report,
        "timestamp_diagnostics",
        problems,
        f"{len(stamps)} stamps checked for numeric signatures",
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
                entry = {}
        out.append(dict(entry) if isinstance(entry, Mapping) else {})
    return out


def _check_annotation_intervals(rule, context, report) -> None:
    total_frames = context.get("total_frames") or 0
    problems = []
    for annotation in _annotations(context, rule.get("key", "annotations")):
        start, end = annotation.get("start_idx"), annotation.get("end_idx")
        if any(not isinstance(v, int) or isinstance(v, bool) for v in (start, end)):
            problems.append("annotation must contain integer start_idx and end_idx")
        elif start < 0 or end > total_frames or end <= start:
            problems.append(
                f"span [{start}, {end}) is outside [0, {total_frames}) or empty"
            )
    _report_problems(
        rule, report, "annotation_intervals", problems, "declared spans are valid"
    )


def _sample_indices(count: int, limit: int) -> np.ndarray:
    """Return all indices if ``limit <= 0``; otherwise sample up to ``limit``.

    Sampling is unnecessary when ``count <= limit``. A sampled result includes
    indices 0 and ``count - 1`` and spaces the remaining integer indices as
    evenly as possible.
    """
    if limit <= 0 or count <= limit:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, limit).astype(int))


def _check_fk_residual(rule, context, report) -> None:
    resolved: ResolvedEmbodiment | None = context.get("resolved")
    if resolved is None:
        return
    arrays = context["arrays"]
    total_frames = context.get("total_frames")
    problems = []
    passed = []
    for side, end_effector in sorted(resolved.end_effectors.items()):
        if end_effector.urdf_path is None or not end_effector.urdf_path.is_file():
            if end_effector.ee_class == "dexterous_hand":
                report.add(
                    WARNING,
                    "fk_unavailable",
                    f"{side}: no available URDF; stored arrays remain usable, FK agreement cannot be checked",
                )
            continue
        tracks = [
            arrays.get(f"{side}.{rule[name]}")
            for name in ("joints_suffix", "keypoints_suffix", "pose_suffix")
        ]
        if any(track is None for track in tracks):
            # Array-presence rules report required tracks separately. The FK
            # check compares only complete joints/keypoints/pose triples.
            continue
        joints, keypoints, poses = (_read(t, total_frames) for t in tracks)
        rows = min(len(joints), len(keypoints), len(poses))
        if rows == 0:
            continue
        frames = _sample_indices(rows, int(rule.get("sample_frames", 64)))
        try:
            residuals = keypoint_residuals(
                end_effector, joints[frames], keypoints[frames], poses[frames]
            )
        except (UrdfError, ValueError, np.linalg.LinAlgError) as exc:
            problems.append(f"{side}: {exc}")
            continue
        worst = float(residuals.max())
        tolerance = float(end_effector.fk_tolerance_m)
        if not np.isfinite(residuals).all():
            problems.append(f"{side}: non-finite FK residuals")
        elif worst > tolerance:
            slot = list(end_effector.keypoints.valid)[
                int(np.unravel_index(residuals.argmax(), residuals.shape)[1])
            ]
            problems.append(
                f"{side}: forward kinematics over {end_effector.urdf} disagrees "
                f"with {side}.{rule['keypoints_suffix']} by {worst:.4f} m at slot "
                f"{slot} (tolerance {tolerance:g} m); check joint order, units, "
                "handedness and the root frame"
            )
        else:
            passed.append(f"{side} within {worst:.4f} m")
    if not problems and not passed:
        return
    _report_problems(rule, report, "fk_residual", problems, "; ".join(passed))


def _check_tactile_declaration(rule, context, report) -> None:
    resolved: ResolvedEmbodiment | None = context.get("resolved")
    if resolved is None:
        return
    suffix = rule.get("suffix", "obs_tactile")
    problems = []
    declared = 0
    for side, end_effector in sorted(resolved.end_effectors.items()):
        if f"{side}.{suffix}" not in context["arrays"]:
            continue
        if end_effector.tactile is None:
            problems.append(
                f"{side}.{suffix} is stored, but end-effector "
                f"{end_effector.name!r} declares no `tactile:` block; a taxel "
                "count without a unit cannot be compared across hands"
            )
        else:
            declared += 1
    if not problems and not declared:
        return
    _report_problems(
        rule,
        report,
        "tactile_declaration",
        problems,
        f"{declared} tactile array(s) match a registry declaration",
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
    "annotation_intervals": _check_annotation_intervals,
    "fk_residual": _check_fk_residual,
    "tactile_declaration": _check_tactile_declaration,
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
    """Resolve a shape token; return ``None`` for a wildcard or absent spec."""
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
    if token == "tactile":
        end_effector = resolved.end_effectors.get(side) if side else None
        if end_effector is None or end_effector.tactile is None:
            return None
        return end_effector.tactile.width
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
        level = _level(rule, report, key)
        if level is not None:
            report.add(level, key, "missing")
        return

    array = arrays[key]
    shape = tuple(int(n) for n in array.shape)
    expected = rule.get("shape")
    problems = []
    if expected is not None:
        if len(shape) != len(expected):
            problems.append(f"expected {len(expected)} dimension(s), got shape {shape}")
        else:
            for axis, token in enumerate(expected):
                want = _dimension(token, context, side)
                if want is None:
                    continue
                # ``total_frames`` is the authoritative length and a stored
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
        _check_numeric_values(key, array, context, report, side)


def _check_numeric_values(key, array, context, report, side):
    """Check retained estimates, respecting Human sentinels and registry slots."""
    if _dtype_kind(array.dtype) != "float":
        return
    values = _read(array, context["total_frames"])
    resolved = context["resolved"]
    spec = resolved.end_effectors.get(side)
    human = resolved.platform.kind == "human"
    if "keypoints" in key and spec is not None:
        values = values.reshape(len(values), spec.keypoints.n_slots, 3)
        values = values[:, spec.keypoints.valid, :]
        # Legacy human estimates may be missing (all NaN or the Aria 1e9
        # sentinel). A partial nonfinite point is still corrupt data.
        missing = np.isnan(values).all(axis=-1) | (np.abs(values) >= 1e8).all(axis=-1)
        if human and missing.any():
            report.add(WARNING, key, f"{missing.sum()} missing human keypoint estimates")
            values = values[~missing]
    if key.endswith("pose") and values.ndim == 2 and values.shape[1] == 7:
        missing = (np.abs(values) >= 1e8).all(axis=-1)
        if human and key != "obs_head_pose" and missing.any():
            report.add(WARNING, key, f"{missing.sum()} missing human pose estimates")
            values = values[~missing]
        norms = np.linalg.norm(values[:, 3:], axis=-1)
        if not np.isfinite(values).all():
            report.add(ERROR, key, "nonfinite retained pose values")
        if not np.all(np.isfinite(norms) & np.isclose(norms, 1, atol=1e-3)):
            report.add(ERROR, key, "retained quaternions must be finite and unit norm (wxyz)")
        return
    if not np.isfinite(values).all():
        report.add(ERROR, key, "nonfinite retained values in valid slots")
    if "keypoints" in key and (np.abs(values) >= 1e8).any():
        report.add(ERROR, key, "invalid retained keypoint estimates in valid slots")
    if key.endswith("hand_joints") and spec is not None and spec.joint_limits:
        limits = np.asarray(spec.joint_limits)
        if ((values < limits[:, 0] - 1e-6) | (values > limits[:, 1] + 1e-6)).any():
            report.add(ERROR, key, "retained joints exceed registry limits; check units and joint order")


def _expand_key(template: str, arrays: Mapping, sides) -> list[tuple[str, str | None]]:
    """Expand a schema key into the concrete array keys it selects.

    ``{side}`` expands over the resolved side candidates. ``*`` selects stored
    keys only, so a wildcard rule never creates a missing-key finding.
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


def validate_episode(
    path: str | Path, *, requirements: Mapping[str, bool] | None = None,
    ego_overlay: bool = False,
) -> Report:
    """Validate one Zarr episode against ``schema/episode_v3.yaml``.

    Args:
        path: The episode ``.zarr`` directory.
        requirements: Per-rule severity decisions. ``True`` reports a failure
            as an error; ``False`` waives it to a warning. Omitted rules remain
            optional. Integrity rules cannot be overridden.
        ego_overlay: Require a usable front_1 keypoint overlay over retained RGB
            frames, with explicit dexterous delivery metadata and joint order.

    Returns:
        The findings emitted while validating the episode.

    Raises:
        SchemaError: If the schema or a requirement override is invalid.
    """
    path = Path(path)
    schema = load_schema()
    report = Report(path=path, requirements=_requirements(schema, requirements))
    report.capabilities["ego_overlay"] = {"requested": ego_overlay, "available": None}
    try:
        _validate_store(path, schema, report, ego_overlay)
    except SchemaError:
        raise
    except Exception as exc:
        # An unreadable array or corrupt episode must not abort a multi-path
        # invocation. Schema configuration errors above remain programmer errors.
        report.add(ERROR, "episode", f"cannot validate episode: {type(exc).__name__}: {exc}")
    return report


def _validate_store(path, schema, report, ego_overlay):
    try:
        store = zarr.open_group(str(path), mode="r")
    except Exception as exc:
        report.add(ERROR, "episode", f"cannot open as a zarr group: {exc}")
        return
    attrs = dict(store.attrs)
    arrays = {name: store[name] for name in store.array_keys()}
    report.data_status = data_status(attrs)
    report.status_eligible = is_complete(attrs)
    context = {
        "store": store,
        "array_keys": list(arrays),
        "arrays": arrays,
        "attrs": attrs,
        "features": attrs.get("features") if isinstance(attrs.get("features"), Mapping) else {},
        "total_frames": attrs.get("total_frames"),
    }
    embodiment = attrs.get("embodiment")
    if isinstance(embodiment, str):
        context["named_platform"] = load_embodiment_platforms().get(
            canonical_embodiment_name(embodiment)
        )
    try:
        context["calibration"] = read_calibration(attrs)
    except (CalibrationError, TypeError, ValueError) as exc:
        context["calibration"] = None
        report.add(ERROR, "calibration_format", str(exc))

    for name, rule in schema.get("attributes", {}).items():
        _check_attribute(name, rule, attrs, report, context)
    # Validate prerequisites before reading tracks or computing dependent checks.
    total = context["total_frames"]
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        if ego_overlay:
            report.capabilities["ego_overlay"]["available"] = False
        return
    resolved = _resolve(attrs, report)
    context["resolved"] = resolved
    if resolved is not None:
        report.add(OK, "embodiment", resolved.describe())
        for rule in schema.get("arrays", []):
            for key, side in _expand_key(rule["key"], arrays, resolved.sides):
                if side is not None and side not in resolved.end_effectors:
                    continue
                when = rule.get("when") or {}
                if when and not _condition_holds(when, resolved, side):
                    continue
                _check_array(key, rule, arrays, report, context, side)

    _check_features(context, report, required=ego_overlay)
    for rule in schema.get("checks", []):
        check = _NAMED_CHECKS.get(rule.get("name"))
        if check is None:
            raise SchemaError(f"unknown check {rule.get('name')!r} in the schema")
        try:
            check(rule, context, report)
        except (TypeError, ValueError, IndexError, KeyError, OverflowError) as exc:
            report.add(ERROR, rule["name"], f"invalid prerequisites: {exc}")
    if ego_overlay:
        _check_ego_overlay(context, report)


def _check_features(context, report, *, required):
    features = context["features"]
    for key, array in context["arrays"].items():
        if key not in features:
            report.add(ERROR if required else WARNING, "features", f"missing description for {key}")
            continue
        feature = features[key]
        if not isinstance(feature, Mapping):
            report.add(ERROR, "features", f"{key}: description must be a mapping")
            continue
        shape = feature.get("shape")
        if not isinstance(shape, (list, tuple)) or not all(
            isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in shape
        ):
            report.add(ERROR, "features", f"{key}: shape must list nonnegative dimensions")
        elif not key.startswith("images.") and key != "annotations" and tuple(shape) != array.shape[1:]:
            report.add(ERROR, "features", f"{key}: described shape {shape} differs from stored {array.shape[1:]}")
        dtype = feature.get("dtype")
        if not isinstance(dtype, str):
            report.add(ERROR, "features", f"{key}: dtype must be a string")
        elif _dtype_kind(array.dtype) in ("float", "int") and dtype != str(array.dtype):
            report.add(ERROR, "features", f"{key}: described dtype {dtype} differs from stored {array.dtype}")


def _check_ego_overlay(context, report):
    from egomimic.rldb.zarr.camera_coverage import camera_coverage
    from egomimic.rldb.zarr.overlay import decode_frame, keypoint_chunk

    store, resolved = context["store"], context["resolved"]
    problems = []
    checked = 0
    if resolved is None:
        problems.append("cannot resolve morphology")
    else:
        dexterous = any(s.ee_class == "dexterous_hand" for s in resolved.end_effectors.values())
        if dexterous:
            if not isinstance(context["attrs"].get("morphology"), Mapping):
                problems.append("dexterous delivery must declare morphology")
            timestamps = context["arrays"].get("obs_rgb_timestamps_ns")
            if timestamps is None or str(timestamps.dtype) != "int64":
                problems.append("dexterous RGB timestamps must use int64 UTC nanoseconds")
            for side, spec in resolved.end_effectors.items():
                if spec.ee_class != "dexterous_hand":
                    continue
                for prefix in ("obs", "cmd"):
                    key = f"{side}.{prefix}_hand_joints"
                    feature = context["features"].get(key)
                    if not isinstance(feature, Mapping) or feature.get("joint_names") != list(spec.joint_names):
                        problems.append(f"{key}: features.joint_names must declare registry joint order")
            if resolved.platform.aux and resolved.platform.aux.joint_names:
                for key in ("obs_aux_joints", "cmd_aux_joints"):
                    feature = context["features"].get(key)
                    if not isinstance(feature, Mapping) or feature.get("joint_names") != list(resolved.platform.aux.joint_names):
                        problems.append(f"{key}: features.joint_names must declare registry auxiliary order")
        calibration = context.get("calibration")
        entry = None if calibration is None else calibration.cameras.get("front_1")
        if entry is None or entry.K is None:
            problems.append("front_1: declared intrinsics required for ego overlay")
        elif dexterous and (calibration.legacy or entry.resolution is None):
            problems.append("front_1: dexterous calibration must declare the stored image resolution")
        if dexterous and not problems:
            camera_block = context["attrs"].get("calibration", {}).get("cameras", {}).get("front_1", {})
            if not all(key in camera_block for key in ("model", "rectified")):
                problems.append("front_1: declare the camera model and rectified image status")
        if not problems:
            for frame in range(context["total_frames"]):
                try:
                    image = decode_frame(store, frame, "front_1")
                    coverage = camera_coverage(store, "front_1", frame, resolved=resolved, image_shape=image.shape)
                    if not coverage.available:
                        raise ValueError("; ".join(coverage.missing))
                    feature = context["features"].get("images.front_1", {})
                    if tuple(feature.get("shape", ())) != image.shape:
                        raise ValueError("images.front_1: feature shape differs from decoded RGB")
                    # Consume the same supplied arrays and validity masks as
                    # the renderer. No FK generation or calibration fitting.
                    _, _, chunk, owned = keypoint_chunk(store, frame)
                    points = chunk.reshape(-1, 3)[owned]
                    if not np.isfinite(points).all():
                        raise ValueError("front_1: missing/invalid keypoints in active slots")
                    checked += 1
                except (ValueError, KeyError, OSError) as exc:
                    problems.append(f"frame {frame}: {exc}")
                    break
    report.capabilities["ego_overlay"] = {
        "requested": True, "available": not problems,
        "checked_frames": checked, "missing": problems,
    }
    report.add(ERROR if problems else OK, "ego_overlay", "; ".join(problems) if problems else f"front_1 RGB, calibration and keypoints usable over {checked} retained frames")


def _resolve(attrs: Mapping, report: Report) -> ResolvedEmbodiment | None:
    """Honor explicit morphology and constrain it to the named active sides."""
    try:
        return Embodiment.from_attrs(attrs)
    except (TypeError, ValueError):
        pass
    report.add(
        ERROR,
        "embodiment",
        "cannot resolve the episode's embodiment, so no array rule can run",
    )
    return None


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser from the schema's waivable rules."""
    parser = argparse.ArgumentParser(
        prog="python -m egomimic.rldb.zarr.validate",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("paths", nargs="+", type=Path, help="episode .zarr paths")
    for name, rule in waivable_rules().items():
        flag = name.replace("_", "-")
        # Escape literal percent signs in schema help text for argparse.
        help_text = rule["why"].replace("%", "%%")
        parser.add_argument(
            f"--{flag}",
            dest=f"require_{name}",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=f"[{rule['severity']}] {help_text} (default: report limitation)",
        )
    parser.add_argument("--ego-overlay", action="store_true",
                        help="Require front_1 RGB, supplied keypoints and optical camera transforms over every retained frame")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="list the rules that passed as well",
    )
    parser.add_argument(
        "--json", action="store_true", help="print a JSON array of episode reports"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate CLI paths and return an exit status.

    Returns:
        ``1`` for validation errors; ``3`` for a valid but ineligible status
        when --data-status is requested; otherwise ``0``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    requirements = {name: getattr(args, f"require_{name}") for name in waivable_rules()}
    reports = [validate_episode(p, requirements=requirements, ego_overlay=args.ego_overlay) for p in args.paths]
    if args.json:
        print(json.dumps([r.to_jsonable() for r in reports], indent=2))
    else:
        for report in reports:
            print(report.text(verbose=args.verbose))
    if not all(r.ok for r in reports):
        return 1
    if args.require_data_status and any(not r.status_eligible for r in reports):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
