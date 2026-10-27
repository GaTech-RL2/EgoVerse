"""
Metadata required in every contributed zarr episode.

See CONTRIBUTING_DATA.md §6.3. ZarrWriter takes each REQUIRED_METADATA_FIELDS
entry as a keyword argument with no default and checks it in write(); the
episode validator (egomimic.rldb.zarr.validate) applies the same rules to a
finished store. task_name, operator and lab are always required; scene and
objects can be left out only with an explicit MissingMetadata().
"""

import argparse
import re
from collections.abc import Iterable
from typing import Any

REQUIRED_METADATA_FIELDS = ("task_name", "operator", "lab", "scene", "objects")
WAIVABLE_FIELDS = ("scene", "objects")

# Values that mean "I didn't fill this in". Rejected so a placeholder can't
# stand in for a real value (compared case-insensitively after strip).
_PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "unknown",
        "none",
        "null",
        "n/a",
        "na",
        "tbd",
        "todo",
        "debug",
        "test",
        "placeholder",
        "template",
        "missing",
        "nil",
        "nan",
        "-",
        "?",
        "...",
    }
)

_LAB_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


def is_placeholder(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDER_VALUES


class MissingMetadata:
    """Explicit opt-out for ``scene`` or ``objects`` when the source data has no value.

    Pass it in place of the value, e.g. ``scene=MissingMetadata()``. task_name,
    operator and lab cannot be waived. ``None``,
    ``""`` and placeholders like ``"unknown"`` are rejected, so a field can only
    be left out on purpose. The field is stored as ``null`` in zarr.attrs.

    Only use this if absolutely necessary, i.e. the source data truly has no
    way to recover the value. Episodes with missing metadata are much harder
    to filter and train on, and fail validation without --allow-missing.
    """

    def __repr__(self) -> str:
        return "MissingMetadata()"


def objects_list(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """``objects`` as a list: a str is a comma-separated list of names."""
    if isinstance(value, str):
        return [o.strip() for o in value.split(",")]
    return list(value)


def required_metadata_error(field: str, value: Any) -> str | None:
    """Return why ``value`` is not acceptable for required metadata ``field``, or None.

    Shared by ZarrWriter.write and the episode validator so both
    apply the same rules. ``value`` is a concrete value, not MissingMetadata.
    """
    hint = "See CONTRIBUTING_DATA.md §6.3."
    if field in WAIVABLE_FIELDS:
        waive = f"If the source data has no {field}, pass {field}=MissingMetadata(). "
    else:
        waive = f"{field} is always required and cannot be waived. "
    if field == "objects" and isinstance(value, (list, tuple, str)):
        if isinstance(value, str) and value != value.strip():
            return f"objects={value!r} has leading/trailing whitespace. {hint}"
        names = objects_list(value)
        if not names or names == [""]:
            return f"objects must name at least one object. {waive}{hint}"
        for obj in names:
            if not isinstance(obj, str) or is_placeholder(obj):
                return f"objects entries must be real names, got {obj!r}. {hint}"
            if obj != obj.strip():
                return f"objects entry {obj!r} has leading/trailing whitespace. {hint}"
            if not obj.isprintable():
                return f"objects entry {obj!r} has non-printable characters. {hint}"
        return None
    if not isinstance(value, str):
        extra = " or list[str]" if field == "objects" else ""
        return (
            f"{field} must be a str{extra}, got {type(value).__name__} ({value!r}). "
            f"{waive}{hint}"
        )
    if is_placeholder(value):
        return f"{field}={value!r} is a placeholder, not a value. {waive}{hint}"
    if value != value.strip():
        return f"{field}={value!r} has leading/trailing whitespace. {hint}"
    if not value.isprintable():
        return f"{field}={value!r} has non-printable characters. {hint}"
    if field == "lab" and not _LAB_RE.fullmatch(value):
        return (
            f"lab={value!r} must be a short, stable, lowercase identifier "
            f"(letters, digits, '_' or '-'), e.g. 'mecka'. {hint}"
        )
    return None


def vendor_recording_id_error(value: Any) -> str | None:
    """Why ``value`` is not a usable vendor_recording_id (§6.6), or None.

    It names the raw recording under s3://rldb/raw/<lab>/<vendor_recording_id>/
    and, with lab, gives the Lance recording_id, so it is one path component.
    """
    hint = "See CONTRIBUTING_DATA.md §6.6."
    if not isinstance(value, str) or is_placeholder(value):
        return (
            "vendor_recording_id must name the raw recording this episode was cut "
            f"from, got {value!r}. {hint}"
        )
    if value != value.strip() or not value.isprintable() or "/" in value:
        return (
            f"vendor_recording_id={value!r} must be a single path component with no "
            f"surrounding whitespace or non-printable characters. {hint}"
        )
    return None


def check_stored_metadata(
    meta: dict[str, Any], *, allow_missing: bool = False
) -> tuple[list[str], list[str], list[str]]:
    """(errors, warnings, passed field names) for a finished episode's metadata.

    Shared by the zarr and MCAP validators. An absent key is always an error; a
    ``None`` value is an explicit waiver (MissingMetadata), an error unless
    ``allow_missing`` and the field is in WAIVABLE_FIELDS. Concrete values go
    through ``required_metadata_error``.
    """
    errors, warnings, passed = [], [], []
    for name in REQUIRED_METADATA_FIELDS:
        if name not in meta:
            errors.append(
                f"Missing metadata field: {name}. See CONTRIBUTING_DATA.md §6.3."
            )
        elif meta[name] is None and name not in WAIVABLE_FIELDS:
            errors.append(
                f"{name} is null: it is always required and cannot be waived. "
                "See CONTRIBUTING_DATA.md §6.3."
            )
        elif meta[name] is None:
            msg = f"{name}: marked as not available (MissingMetadata)"
            if allow_missing:
                warnings.append(msg)
            else:
                errors.append(
                    msg + ". We heavily recommend providing it; pass "
                    "--allow-missing to accept the episode anyway."
                )
        elif (err := required_metadata_error(name, meta[name])) is not None:
            errors.append(err)
        else:
            passed.append(name)
    return errors, warnings, passed


def add_required_metadata_args(parser: argparse.ArgumentParser) -> None:
    """Add --operator/--lab/--scene/--objects/--missing-metadata to a converter CLI.

    The converter supplies its own --task-name. None of these have usable
    defaults: an omitted field fails in ZarrWriter.write unless it is scene or
    objects waived with ``--missing-metadata FIELD``.
    """
    group = parser.add_argument_group(
        "required episode metadata (CONTRIBUTING_DATA.md §6.3)"
    )
    group.add_argument("--operator", type=str, help="ID of the collecting operator")
    group.add_argument("--lab", type=str, help="Short lowercase source id, e.g. rl2")
    group.add_argument("--scene", type=str, help="Scene / location label")
    group.add_argument("--objects", type=str, help="Objects manipulated")
    group.add_argument(
        "--missing-metadata",
        action="append",
        default=[],
        choices=WAIVABLE_FIELDS,
        metavar="FIELD",
        help="Explicitly mark FIELD (scene or objects) as not available in the "
        "source data. "
        "Repeatable. Only use this if absolutely necessary: provide the value "
        "whenever the source data has it.",
    )


def _build_required_metadata(
    values: dict[str, Any], missing: Iterable[str] = ()
) -> dict[str, Any]:
    """Merge plain values and the names of waived fields into ZarrWriter kwargs.

    Both inputs are plain data so they can travel through Ray / argparse.
    A field named in ``missing`` becomes MissingMetadata(); giving a field both
    a value and a waiver is an error.
    """
    if errors := waiver_errors(values, missing):
        raise ValueError(errors[0])
    missing = set(missing)
    return {
        field: MissingMetadata() if field in missing else values.get(field)
        for field in REQUIRED_METADATA_FIELDS
    }


def waiver_errors(values: dict[str, Any], missing: Iterable[str]) -> list[str]:
    """Waived fields that are unknown or not waivable, and fields with both a
    value and a waiver."""
    missing = set(missing)
    errors = []
    unknown = missing - set(REQUIRED_METADATA_FIELDS)
    if unknown:
        errors.append(
            f"missing_metadata names unknown fields {sorted(unknown)}; "
            f"expected a subset of {list(WAIVABLE_FIELDS)}"
        )
    if unwaivable := sorted(
        (missing & set(REQUIRED_METADATA_FIELDS)) - set(WAIVABLE_FIELDS)
    ):
        errors.append(
            f"{unwaivable} cannot be waived; only {list(WAIVABLE_FIELDS)} can be "
            "marked missing. See CONTRIBUTING_DATA.md §6.3."
        )
    for field in REQUIRED_METADATA_FIELDS:
        value = values.get(field)
        empty = value is None or (hasattr(value, "__len__") and len(value) == 0)
        if field in missing and not empty:
            errors.append(
                f"{field} has both a value ({value!r}) and is marked missing; "
                "pass one or the other"
            )
    return errors


def check_writer_metadata(kwargs: dict[str, Any]) -> None:
    """Raise ValueError on the first unacceptable value in ZarrWriter metadata kwargs.

    The same check ZarrWriter.write runs; call it before an expensive extraction
    so a bad value fails fast.
    """
    for field in REQUIRED_METADATA_FIELDS:
        value = kwargs.get(field)
        if isinstance(value, MissingMetadata):
            if field in WAIVABLE_FIELDS:
                continue
            raise ValueError(
                f"{field}=MissingMetadata(): {field} is always required and "
                "cannot be waived. See CONTRIBUTING_DATA.md §6.3."
            )
        if (err := required_metadata_error(field, value)) is not None:
            raise ValueError(err)


def required_metadata_from_args(args: Any) -> dict[str, Any]:
    """ZarrWriter required-metadata kwargs from an argparse/SimpleNamespace.

    ``args.missing_metadata`` is an iterable of waived field names (repeated
    --missing-metadata on the CLI, or a list from run_conversion).
    """
    missing = getattr(args, "missing_metadata", None) or ()
    values = {f: getattr(args, f, None) for f in REQUIRED_METADATA_FIELDS}
    return _build_required_metadata(values, missing)
