"""What a delivery is, decided from its files, and how to prepare it.

Deliveries land under deliveries/<vendor>/<delivery_id>/ without declaring a format. A
format recognizes its deliveries by file-name patterns (`{id}` is one path component that
every file of a delivery must agree on, the recording's vendor id; `*` and `**` glob
within a component and across them), says which files it accepts, and supplies
its prepare (egomimic/ingest/stage.py shape), which turns a delivery into a staged
delivery the committer (egomimic.scripts.lance_commit) can take as it is. The conversion
job calls prepare_delivery and never needs to know which format it is handling; a
delivery no format recognizes is `unsupported`, not guessed at.

    mcap   episode.mcap (+ optional annotation.mcap): vendor MCAP spec 0.1
    eva    <id>.hdf5 + <id>_metadata.json: one EVA demo (egomimic/ingest/eva.py)
    aria   <id>.vrs + <id>_metadata.json + mps_<id>_vrs/: one Aria recording
           (egomimic/ingest/aria.py)
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from egomimic.ingest.stage import Prepared


@cache
def _pattern(p: str) -> re.Pattern:
    out = ""
    for tok in re.split(r"(\{id\}|\*\*|\*)", p):
        if tok == "{id}":
            out += r"(?P<id>[^/]+)"
        elif tok == "**":
            out += r".*"
        elif tok == "*":
            out += r"[^/]*"
        else:
            out += re.escape(tok)
    return re.compile(out)


def _ids(patterns: Iterable[str], name: str) -> set[str | None]:
    """The `{id}` values (None for a pattern without one) under which `name` matches."""
    out = set()
    for p in patterns:
        if m := _pattern(p).fullmatch(name):
            out.add(m.groupdict().get("id"))
    return out


@dataclass(frozen=True)
class Format:
    name: str
    description: str
    required: frozenset[str]  # file patterns every delivery of this format has
    optional: frozenset[str]  # ones it may also have; anything else is rejected
    max_file_bytes: int  # per file: checked by the dashboard, the watcher and prepare
    prepare: Callable[
        ..., Prepared
    ]  # (delivery, staged, lance_root, *, fingerprint, vendor, ...)
    # What its prepare takes beyond the delivery: conversion settings
    # (convert_job.convert_settings names) and per-vendor defaults (default_license).
    settings: frozenset[str] = frozenset()

    def recording_key(self, names: Iterable[str]) -> str | None:
        """The one `{id}` all of the required patterns match under, else None (for
        formats without `{id}`: "" when every required pattern matches)."""
        names = set(names)
        common = None
        for p in self.required:
            ids = set().union(*(_ids([p], n) for n in names)) if names else set()
            common = ids if common is None else common & ids
        if not common:
            return None
        if len(common) > 1:  # several recordings in one delivery
            return None
        (key,) = common
        return key or ""

    def matches(self, names: set[str]) -> bool:
        return self.recording_key(names) is not None

    def unexpected(self, names: set[str]) -> list[str]:
        """Files no pattern accepts, or accepted only under another recording's id."""
        key = self.recording_key(names)
        return sorted(
            n
            for n in names
            if not (_ids(self.allowed, n) & ({key, None} if key else {None, ""}))
        )

    def accepts(self, name: str) -> bool:
        """Whether some file of some delivery of this format may be called `name`
        (before the rest of the delivery is known, e.g. when its upload starts)."""
        return bool(_ids(self.allowed, name))

    @property
    def allowed(self) -> frozenset[str]:
        return self.required | self.optional

    def oversize(self, sizes: dict[str, int]) -> list[str]:
        """Problems for files over the limit; `sizes` maps name -> bytes, e.g. from the
        bucket listing, so the watcher can refuse before downloading."""
        return [
            f"{name} is {size / 1024**3:.1f} GiB, over the {self.name} limit of "
            f"{self.max_file_bytes / 1024**3:.0f} GiB"
            for name, size in sorted(sizes.items())
            if size > self.max_file_bytes
        ]


def _mcap() -> Format:
    from egomimic.ingest import stage
    from egomimic.rldb.mcap import spec

    return Format(
        name="mcap",
        description="vendor MCAP spec 0.1 (egomimic/rldb/mcap/SPEC.md)",
        required=frozenset({spec.EPISODE_FILE}),
        optional=frozenset({spec.ANNOTATION_FILE}),
        max_file_bytes=200 * 1024**3,  # a long multi-camera recording
        prepare=stage.prepare,
        settings=frozenset({"codec", "ffmpeg", "mano", "mano_iters", "mano_model_dir"}),
    )


def _eva() -> Format:
    from egomimic.ingest import eva

    return Format(
        name="eva",
        description="one EVA demo: <id>.hdf5 and <id>_metadata.json, <id> its start "
        "in epoch ms (egomimic/ingest/eva.py)",
        required=frozenset(eva.FILES),
        optional=frozenset(),
        max_file_bytes=200 * 1024**3,  # raw HDF5 images are uncompressed
        prepare=eva.prepare,
        settings=frozenset({"ffmpeg", "default_license"}),
    )


def _aria() -> Format:
    from egomimic.ingest import aria

    return Format(
        name="aria",
        description="one Aria recording: <id>.vrs, <id>_metadata.json and its MPS "
        "outputs in mps_<id>_vrs/, <id> its start in epoch ms (egomimic/ingest/aria.py)",
        required=frozenset(aria.FILES),
        optional=frozenset(aria.OPTIONAL),
        max_file_bytes=200 * 1024**3,
        prepare=aria.prepare,
        settings=frozenset(
            {"ffmpeg", "mano", "mano_iters", "mano_model_dir", "default_license"}
        ),
    )


def formats() -> list[Format]:
    """Every known format. Add new ones here."""
    return [_mcap(), _eva(), _aria()]


def accepting(name: str) -> list[Format]:
    """The formats a file called `name` could belong to."""
    return [f for f in formats() if f.accepts(name)]


def by_name(name: str) -> Format:
    for f in formats():
        if f.name == name:
            return f
    raise KeyError(f"unknown delivery format {name!r}")


class AmbiguousFormat(Exception):
    """More than one format claims the same delivery: a bug in the registry."""


def detect(names: Iterable[str]) -> Format | None:
    """The one format whose required files the delivery has, or None."""
    names = set(names)
    found = [f for f in formats() if f.matches(names)]
    if len(found) > 1:
        raise AmbiguousFormat(f"{sorted(names)} matches {[f.name for f in found]}")
    return found[0] if found else None


def _sizes(delivery: Path) -> dict[str, int]:
    return {
        p.relative_to(delivery).as_posix(): p.stat().st_size
        for p in Path(delivery).rglob("*")
        if p.is_file()
    }


def prepare_delivery(
    delivery: Path,
    staged: Path,
    lance_root: Path,
    *,
    fingerprint: str,
    vendor: str,
    **settings,
) -> Prepared:
    """Recognize a downloaded delivery and prepare it with its format. Unrecognized
    deliveries are `unsupported`; recognized ones with extra or oversized files are
    `rejected`, before anything is parsed (internal tooling can bypass the dashboard's
    upload checks)."""
    sizes = _sizes(delivery)
    names = set(sizes)
    fmt = detect(names)
    if fmt is None:
        known = "; ".join(
            f"{f.name}: {', '.join(sorted(f.required))}" for f in formats()
        )
        return Prepared(
            "unsupported",
            problems=[f"no known format has files {sorted(names)} (known: {known})"],
        )
    problems = fmt.oversize(sizes)
    extra = fmt.unexpected(names)
    if extra:
        problems.insert(
            0,
            f"{fmt.name} deliveries hold only {sorted(fmt.allowed)}; "
            f"unexpected: {extra}",
        )
    if problems:
        return Prepared("rejected", problems=problems)
    return fmt.prepare(
        delivery,
        staged,
        lance_root,
        fingerprint=fingerprint,
        vendor=vendor,
        **{k: v for k, v in settings.items() if k in fmt.settings},
    )
