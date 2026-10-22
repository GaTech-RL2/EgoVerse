"""What a delivery is, decided from its files, and how to prepare and commit it.

Deliveries land under deliveries/<vendor>/<delivery_id>/ without declaring a format. A
format recognizes its deliveries by file names, says which files it accepts, and supplies
the prepare / commit pair (egomimic/ingest/stage.py shape). The watcher calls
prepare_delivery and commit_delivery and never needs to know which format it is
handling; a delivery no format recognizes is `unsupported`, not guessed at.

    mcap   episode.mcap (+ optional annotation.mcap): vendor MCAP spec 0.1
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from egomimic.ingest.stage import Prepared


@dataclass(frozen=True)
class Format:
    name: str
    description: str
    required: frozenset[str]  # files every delivery of this format has
    optional: frozenset[str]  # files it may also have; anything else is rejected
    max_file_bytes: int  # per file: checked by the dashboard, the watcher and prepare
    prepare: Callable[
        ..., Prepared
    ]  # (delivery, staged, lance_root, *, fingerprint, ...)
    commit: Callable[..., dict]  # (staged, lance_root, *, approved_stage, fingerprint)

    def matches(self, names: set[str]) -> bool:
        return self.required <= names

    def unexpected(self, names: set[str]) -> list[str]:
        return sorted(names - self.required - self.optional)

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
        commit=stage.commit_staged,
    )


def formats() -> list[Format]:
    """Every known format. Add new ones here."""
    return [_mcap()]


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
    delivery: Path, staged: Path, lance_root: Path, *, fingerprint: str, **settings
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
        delivery, staged, lance_root, fingerprint=fingerprint, **settings
    )


def commit_delivery(
    staged: Path,
    lance_root: Path,
    *,
    approved_stage: str,
    fingerprint: str | None = None,
) -> dict:
    """Commit an approved staged result with the format that staged it (its pin)."""
    from egomimic.ingest.stage import read_pin

    fmt = by_name(read_pin(staged)["format"])
    return fmt.commit(
        staged, lance_root, approved_stage=approved_stage, fingerprint=fingerprint
    )
