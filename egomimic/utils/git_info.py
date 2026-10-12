"""Git facts about the work tree that contains this file (with an editable
install, the egomimic checkout). Output is raw bytes: diffs of non-UTF-8 files
must neither crash nor be re-encoded into something ``git apply`` rejects."""

from __future__ import annotations

import subprocess
from pathlib import Path

# Any path inside the checkout; git walks up to the work tree root itself.
_CWD = Path(__file__).resolve().parent
# stage-runtime.sh runs a node-local copy of the code, which has no .git; it
# records these facts about the checkout when it takes the copy.
_SNAPSHOT = _CWD.parents[1] / ".git_provenance"
# stage-runtime.sh runs the same commands when it records a snapshot.
DIFF_ARGS = ("diff", "HEAD", "--submodule=diff", "--no-ext-diff", "--no-color")
UNTRACKED_ARGS = ("ls-files", "--others", "--exclude-standard")
# Bounds every git call (a large dirty checkout on loaded NFS can be slow).
_TIMEOUT_S = 30.0


class GitError(RuntimeError):
    """git is missing, timed out, or exited non-zero."""


def _snapshot(name: str) -> bytes | None:
    """``name`` as recorded by stage-runtime.sh, or None when not running from a
    staged copy. Raises GitError if git failed when the copy was taken."""
    if not _SNAPSHOT.is_dir():
        return None
    err = _SNAPSHOT / f"{name}.error"
    if err.exists():
        raise GitError(err.read_text(errors="replace").strip())
    return (_SNAPSHOT / name).read_bytes()


def _git(*args: str) -> bytes:
    cmd = ["git", *args]
    try:
        out = subprocess.run(cmd, cwd=_CWD, capture_output=True, timeout=_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as e:
        raise GitError(f"`{' '.join(cmd)}` failed: {e}") from e
    if out.returncode != 0:
        err = out.stderr.decode(errors="replace").strip()
        raise GitError(f"`{' '.join(cmd)}` exited {out.returncode}: {err}")
    return out.stdout


def git_sha() -> str | None:
    """HEAD commit, or None (no git, not inside a work tree, or git failed)."""
    try:
        raw = _snapshot("sha")
        sha = (
            (raw if raw is not None else _git("rev-parse", "HEAD"))
            .decode(errors="replace")
            .strip()
        )
    except GitError:
        return None
    return sha if len(sha) == 40 else None


def git_diff() -> bytes:
    """``git diff HEAD``: staged + unstaged tracked changes, including file
    diffs inside submodules (external/openpi is installed editable). ``b""``
    when clean; raises GitError rather than passing a failure off as clean."""
    raw = _snapshot("diff")
    if raw is not None:
        return raw
    return _git(*DIFF_ARGS)


def git_untracked() -> bytes:
    """Newline-separated paths of untracked, non-ignored files (names only;
    untracked files inside submodules are not listed). Raises GitError."""
    raw = _snapshot("untracked")
    if raw is not None:
        return raw
    return _git(*UNTRACKED_ARGS)
