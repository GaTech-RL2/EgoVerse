"""Git facts about the work tree that contains this file (with an editable
install, the egomimic checkout). Output is raw bytes: diffs of non-UTF-8 files
must neither crash nor be re-encoded into something ``git apply`` rejects."""

from __future__ import annotations

import subprocess
from pathlib import Path

# Any path inside the checkout; git walks up to the work tree root itself.
_CWD = Path(__file__).resolve().parent
# Bounds every git call (a large dirty checkout on loaded NFS can be slow).
_TIMEOUT_S = 30.0


class GitError(RuntimeError):
    """git is missing, timed out, or exited non-zero."""


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
        sha = _git("rev-parse", "HEAD").decode(errors="replace").strip()
    except GitError:
        return None
    return sha if len(sha) == 40 else None


def git_diff() -> bytes:
    """``git diff HEAD``: staged + unstaged tracked changes, including file
    diffs inside submodules (external/openpi is installed editable). ``b""``
    when clean; raises GitError rather than passing a failure off as clean."""
    return _git("diff", "HEAD", "--submodule=diff", "--no-ext-diff", "--no-color")


def git_untracked() -> bytes:
    """Newline-separated paths of untracked, non-ignored files (names only;
    untracked files inside submodules are not listed). Raises GitError."""
    return _git("ls-files", "--others", "--exclude-standard")
