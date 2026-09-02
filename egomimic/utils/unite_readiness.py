"""Fail-closed source identity for the UNITE smoke-to-full transition."""

from __future__ import annotations

import copy
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

MANIFEST_RELATIVE_PATH = "unite_usocket_register_sweep_manifest.yaml"
HANDOFF_RELATIVE_PATH = "UNITE_REGISTER_SWEEP_HANDOFF.md"
SMOKE_BLOCKER = "no_row_has_passed_the_required_real_optimizer_validation_reload_smoke"
_ALLOWED_READINESS_PATHS = frozenset(
    {MANIFEST_RELATIVE_PATH, HANDOFF_RELATIVE_PATH}
)
_READINESS_ONLY_MANIFEST_KEYS = frozenset(
    {"artifact_status", "launch_status", "known_blockers"}
)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class ReadinessTransitionError(RuntimeError):
    """Raised when full training is not source-identical to its passing smoke."""


@dataclass(frozen=True)
class ReadinessIdentity:
    """Validated relationship between a row smoke and a full-run source."""

    smoke_source_head: str
    full_source_head: str
    relation: str
    changed_paths: tuple[str, ...]


def _git(repo: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReadinessTransitionError(
            f"git {' '.join(arguments)} failed with {result.returncode}: {message}"
        )
    return result.stdout


def _validated_commit(repo: Path, value: str, label: str) -> str:
    commit = str(value).strip().lower()
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise ReadinessTransitionError(f"{label} is not a full SHA-1 commit: {value!r}")
    resolved = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved.decode().strip() != commit:
        raise ReadinessTransitionError(f"{label} did not resolve exactly: {commit}")
    return commit


def _manifest_at(repo: Path, commit: str) -> dict:
    raw = _git(repo, "show", f"{commit}:{MANIFEST_RELATIVE_PATH}")
    document = yaml.safe_load(raw)
    if not isinstance(document, dict):
        raise ReadinessTransitionError(
            f"{MANIFEST_RELATIVE_PATH} at {commit} is not a mapping"
        )
    return document


def _require_ready_manifest(document: dict, label: str) -> None:
    if document.get("artifact_status") != "LAUNCH_READY":
        raise ReadinessTransitionError(f"{label} artifact_status is not LAUNCH_READY")
    if document.get("launch_status") != "READY":
        raise ReadinessTransitionError(f"{label} launch_status is not READY")
    if document.get("known_blockers") != []:
        raise ReadinessTransitionError(f"{label} known_blockers is not an empty list")


def _without_readiness(document: dict) -> dict:
    result = copy.deepcopy(document)
    for key in _READINESS_ONLY_MANIFEST_KEYS:
        if key not in result:
            raise ReadinessTransitionError(f"manifest is missing required key {key!r}")
        result.pop(key)
    return result


def verify_readiness_transition(
    repo: Path | str,
    smoke_source_head: str,
    full_source_head: str,
) -> ReadinessIdentity:
    """Prove that a full-run commit is smoke-identical except for readiness docs.

    Exact-head smoke evidence remains valid. A descendant is accepted only when
    the net tree diff is limited to the sweep manifest and handoff, the manifest
    changes exactly from the pre-smoke statuses to the ready statuses, the sole
    smoke blocker is removed, and every other manifest value is unchanged.
    """

    repository = Path(repo).resolve(strict=True)
    top_level = Path(_git(repository, "rev-parse", "--show-toplevel").decode().strip())
    if top_level.resolve() != repository:
        raise ReadinessTransitionError(
            f"repository must be the exact worktree root: {repository} != {top_level}"
        )
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ReadinessTransitionError("repository worktree is not clean")

    smoke_head = _validated_commit(repository, smoke_source_head, "smoke source")
    full_head = _validated_commit(repository, full_source_head, "full source")
    actual_head = _git(repository, "rev-parse", "HEAD").decode().strip()
    if actual_head != full_head:
        raise ReadinessTransitionError(
            f"full source is not current HEAD: {full_head} != {actual_head}"
        )

    current_manifest = _manifest_at(repository, full_head)
    _require_ready_manifest(current_manifest, "full manifest")
    if smoke_head == full_head:
        return ReadinessIdentity(
            smoke_source_head=smoke_head,
            full_source_head=full_head,
            relation="exact_head",
            changed_paths=(),
        )

    try:
        _git(repository, "merge-base", "--is-ancestor", smoke_head, full_head)
    except ReadinessTransitionError as error:
        raise ReadinessTransitionError(
            "smoke source is not an ancestor of the full source"
        ) from error

    changed_raw = _git(
        repository,
        "diff",
        "--name-only",
        "--diff-filter=ACDMRTUXB",
        "-z",
        smoke_head,
        full_head,
        "--",
    )
    changed_paths = tuple(
        sorted(
            path.decode("utf-8")
            for path in changed_raw.split(b"\0")
            if path
        )
    )
    unexpected = set(changed_paths) - _ALLOWED_READINESS_PATHS
    if unexpected:
        raise ReadinessTransitionError(
            f"readiness descendant changes non-readiness paths: {sorted(unexpected)}"
        )
    if MANIFEST_RELATIVE_PATH not in changed_paths:
        raise ReadinessTransitionError("readiness descendant does not change the manifest")

    smoke_manifest = _manifest_at(repository, smoke_head)
    if smoke_manifest.get("artifact_status") != "SMOKE_READY":
        raise ReadinessTransitionError("smoke manifest artifact_status is not SMOKE_READY")
    if smoke_manifest.get("launch_status") != "SMOKE_REQUIRED":
        raise ReadinessTransitionError("smoke manifest launch_status is not SMOKE_REQUIRED")
    if smoke_manifest.get("known_blockers") != [SMOKE_BLOCKER]:
        raise ReadinessTransitionError(
            "smoke manifest does not contain the exact sole smoke blocker"
        )
    if _without_readiness(smoke_manifest) != _without_readiness(current_manifest):
        raise ReadinessTransitionError(
            "manifest changed beyond readiness statuses and the sole smoke blocker"
        )

    return ReadinessIdentity(
        smoke_source_head=smoke_head,
        full_source_head=full_head,
        relation="readiness_only_descendant",
        changed_paths=changed_paths,
    )
