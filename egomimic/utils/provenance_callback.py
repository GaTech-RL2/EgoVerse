"""Record which code produced a run, next to its checkpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from lightning import Callback

from egomimic.utils.git_info import GitError, git_diff, git_sha, git_untracked

log = logging.getLogger(__name__)


class GitProvenance(Callback):
    """Write ``git_sha.txt``, ``git.diff`` and ``git_untracked.txt`` into
    ``trainer.default_root_dir``, once per process: on fit start, or on
    validation start for validate-only runs. Rank 0 only.

    A process that finds ``git_sha.txt`` already there (a Slurm requeue, or any
    resume into the same run dir) writes ``git_sha.<UTC time>.txt`` and so on
    instead, so the launch that trained the earlier checkpoints keeps its
    record. If ``git diff`` or ``git ls-files`` fails, ``<file>.error`` holds
    the reason in place of the file, so a failure never reads as a clean tree.

    wandb records the sha when it is on; this also covers offline runs and
    keeps provenance beside the checkpoints. Never stops a run: without git or a
    checkout it warns and writes nothing.
    """

    def __init__(self) -> None:
        self._done = False

    def on_fit_start(self, trainer, pl_module) -> None:
        self._write_once(trainer)

    def on_validation_start(self, trainer, pl_module) -> None:
        # Validate-only (eval) runs never reach on_fit_start.
        self._write_once(trainer)

    def _write_once(self, trainer) -> None:
        if self._done:
            return
        self._done = True
        if not trainer.is_global_zero:
            return
        root = Path(trainer.default_root_dir)
        try:
            self._write(root)
        except OSError as e:
            log.warning("git provenance: could not write to %s: %s", root, e)

    @staticmethod
    def _write(root: Path) -> None:
        sha = git_sha()
        if sha is None:
            log.warning(
                "git provenance: cannot read HEAD (egomimic is not a git checkout, "
                "or git is missing or failing); not writing git_sha.txt / git.diff"
            )
            return
        root.mkdir(parents=True, exist_ok=True)
        tag = ""
        if (root / "git_sha.txt").exists():
            tag = "." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (root / f"git_sha{tag}.txt").write_text(sha + "\n", encoding="utf-8")
        for name, read in (
            (f"git{tag}.diff", git_diff),
            (f"git_untracked{tag}.txt", git_untracked),
        ):
            try:
                data = read()
            except GitError as e:
                log.warning("git provenance: %s; writing %s.error instead", e, name)
                (root / f"{name}.error").write_text(f"{e}\n", encoding="utf-8")
                continue
            # Bytes as git produced them, so the diff stays applyable.
            (root / name).write_bytes(data)
        log.info("git provenance: %s -> %s/git_sha%s.txt", sha[:12], root, tag)
