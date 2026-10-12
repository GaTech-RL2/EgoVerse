"""Every run dir gets git_sha.txt + git.diff + git_untracked.txt on fit start
(rank 0), and the callback is a no-op with a warning when egomimic is not a git
checkout."""

import locale
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import pytest
import torch

from egomimic.utils import git_info, provenance_callback
from egomimic.utils.git_info import GitError
from egomimic.utils.provenance_callback import GitProvenance


def test_git_helpers_survive_missing_git_binary(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert git_info.git_sha() is None
    with pytest.raises(GitError):
        git_info.git_diff()


def test_failed_git_diff_raises_instead_of_reading_as_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(git_info, "_CWD", tmp_path)  # not a work tree
    assert git_info.git_sha() is None
    with pytest.raises(GitError, match="exited"):
        git_info.git_diff()


def _trainer(tmp_path, rank0=True):
    return SimpleNamespace(default_root_dir=str(tmp_path), is_global_zero=rank0)


def _fake_git(monkeypatch, sha, diff=b"", untracked=b""):
    monkeypatch.setattr(provenance_callback, "git_sha", lambda: sha)
    monkeypatch.setattr(provenance_callback, "git_diff", lambda: diff)
    monkeypatch.setattr(provenance_callback, "git_untracked", lambda: untracked)


def test_callback_writes_sha_and_diff(tmp_path, monkeypatch):
    _fake_git(monkeypatch, "a" * 40, diff=b"+++ b/x\n", untracked=b"new.py\n")
    GitProvenance().on_fit_start(_trainer(tmp_path / "run"), pl_module=None)
    assert (tmp_path / "run" / "git_sha.txt").read_text() == "a" * 40 + "\n"
    assert (tmp_path / "run" / "git.diff").read_bytes() == b"+++ b/x\n"
    assert (tmp_path / "run" / "git_untracked.txt").read_bytes() == b"new.py\n"


def test_failed_diff_is_recorded_not_written_as_clean(tmp_path, monkeypatch, caplog):
    _fake_git(monkeypatch, "a" * 40)

    def fail():
        raise GitError("`git diff HEAD` timed out")

    monkeypatch.setattr(provenance_callback, "git_diff", fail)
    with caplog.at_level("WARNING"):
        cb = GitProvenance()
        cb.on_fit_start(_trainer(tmp_path), pl_module=None)
        cb.on_validation_start(_trainer(tmp_path), pl_module=None)
    assert not (tmp_path / "git.diff").exists()
    assert "timed out" in (tmp_path / "git.diff.error").read_text()
    assert (tmp_path / "git_untracked.txt").exists()
    assert caplog.text.count("timed out") == 1


def test_resume_into_same_run_dir_keeps_original_provenance(tmp_path, monkeypatch):
    """A Slurm requeue reuses the run dir; the first launch's record survives."""
    _fake_git(monkeypatch, "a" * 40, diff=b"first\n")
    GitProvenance().on_fit_start(_trainer(tmp_path), pl_module=None)
    _fake_git(monkeypatch, "b" * 40, diff=b"second\n")
    GitProvenance().on_fit_start(_trainer(tmp_path), pl_module=None)

    assert (tmp_path / "git_sha.txt").read_text() == "a" * 40 + "\n"
    assert (tmp_path / "git.diff").read_bytes() == b"first\n"
    (later_sha,) = tmp_path.glob("git_sha.*Z.txt")
    (later_diff,) = tmp_path.glob("git.*Z.diff")
    assert later_sha.read_text() == "b" * 40 + "\n"
    assert later_diff.read_bytes() == b"second\n"
    assert len(list(tmp_path.glob("git_untracked.*Z.txt"))) == 1


def test_unwritable_run_dir_does_not_stop_training(tmp_path, monkeypatch, caplog):
    _fake_git(monkeypatch, "a" * 40)
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("")
    with caplog.at_level("WARNING"):
        GitProvenance().on_fit_start(_trainer(not_a_dir / "run"), pl_module=None)
    assert "could not write" in caplog.text


def test_callback_is_noop_without_checkout(tmp_path, monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(provenance_callback, "git_sha", lambda: calls.append(1))
    with caplog.at_level("WARNING"):
        cb = GitProvenance()
        cb.on_fit_start(_trainer(tmp_path), pl_module=None)
        for _ in range(3):  # sanity check + validation epochs
            cb.on_validation_start(_trainer(tmp_path), pl_module=None)
    assert not (tmp_path / "git_sha.txt").exists()
    assert len(calls) == 1
    assert len([r for r in caplog.records if "git provenance" in r.message]) == 1


def test_callback_only_writes_on_rank_zero(tmp_path, monkeypatch):
    _fake_git(monkeypatch, "b" * 40)
    GitProvenance().on_fit_start(_trainer(tmp_path, rank0=False), pl_module=None)
    assert not (tmp_path / "git_sha.txt").exists()


def test_default_callbacks_include_git_provenance(compose_resolve):
    from egomimic.utils.instantiators import instantiate_callbacks

    cfg = compose_resolve("train_zarr_cartesian", ["callbacks=checkpoints"])
    kinds = {type(cb).__name__ for cb in instantiate_callbacks(cfg.callbacks)}
    assert "GitProvenance" in kinds


def test_git_sha_matches_rev_parse_in_a_checkout():
    """In a checkout (this worktree, CI's editable install) the helper must
    return HEAD, not merely something 40 characters long."""
    probe = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(git_info.__file__).parent,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("egomimic is not inside a git checkout here")
    assert git_info.git_sha() == probe.stdout.strip()
    assert isinstance(git_info.git_diff(), bytes)


def _run(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway work tree that git_info reports on, isolated from user config."""
    for k, v in {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }.items():
        monkeypatch.setenv(k, v)
    root = tmp_path / "repo"
    root.mkdir()
    _run(root, "init", "-q")
    (root / "mod.py").write_bytes("caf\u00e9 = 1\n".encode())
    (root / ".gitignore").write_text("*.log\n")
    _run(root, "add", ".")
    _run(root, "commit", "-qm", "init")
    monkeypatch.setattr(git_info, "_CWD", root)
    return root


def test_non_utf8_diff_is_written_byte_exact_and_applies(repo, tmp_path, monkeypatch):
    # Simulate the usual UTF-8 locale (text-mode decoding used to raise there).
    monkeypatch.setattr(locale, "getencoding", lambda: "utf-8")
    edited = b"caf\xe9 = 2\n"  # Latin-1, not valid UTF-8
    (repo / "mod.py").write_bytes(edited)
    run = tmp_path / "run"
    GitProvenance().on_fit_start(_trainer(run), pl_module=None)

    diff = (run / "git.diff").read_bytes()
    assert b"+caf\xe9 = 2" in diff
    _run(repo, "checkout", "--", "mod.py")
    _run(repo, "apply", str(run / "git.diff"))
    assert (repo / "mod.py").read_bytes() == edited


def test_untracked_files_are_listed_by_name(repo, tmp_path):
    (repo / "new_module.py").write_text("x = 1\n")
    (repo / "noise.log").write_text("ignored\n")
    GitProvenance().on_fit_start(_trainer(tmp_path / "run"), pl_module=None)
    untracked = (tmp_path / "run" / "git_untracked.txt").read_text().splitlines()
    assert untracked == ["new_module.py"]
    assert b"new_module" not in (tmp_path / "run" / "git.diff").read_bytes()


def test_diff_includes_file_changes_inside_submodules(repo, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _run(sub, "init", "-q")
    (sub / "pi0.py").write_text("layers = 1\n")
    _run(sub, "add", ".")
    _run(sub, "commit", "-qm", "sub")
    allow_file = ("-c", "protocol.file.allow=always")
    _run(repo, *allow_file, "submodule", "add", "-q", str(sub), "ext")
    _run(repo, "commit", "-qm", "add submodule")

    (repo / "ext" / "pi0.py").write_text("layers = 2\n")
    diff = git_info.git_diff()
    assert b"ext/pi0.py" in diff
    assert b"+layers = 2" in diff


class _Tiny(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        return self.net(batch[0]).pow(2).mean()

    def validation_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _loader():
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.zeros(4, 2)), batch_size=2
    )


def test_callback_writes_through_a_real_trainer_on_fit(tmp_path, monkeypatch):
    _fake_git(monkeypatch, "c" * 40, diff=b"diff\n")
    trainer = L.Trainer(
        default_root_dir=str(tmp_path),
        max_steps=1,
        limit_train_batches=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[GitProvenance()],
    )
    trainer.fit(_Tiny(), train_dataloaders=_loader())
    assert (tmp_path / "git_sha.txt").read_text() == "c" * 40 + "\n"
    assert (tmp_path / "git.diff").read_bytes() == b"diff\n"


def test_callback_writes_on_validate_only_runs(tmp_path, monkeypatch):
    """Eval-mode runs call trainer.validate, never fit; they get provenance too."""
    _fake_git(monkeypatch, "d" * 40)
    trainer = L.Trainer(
        default_root_dir=str(tmp_path),
        limit_val_batches=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[GitProvenance()],
    )
    trainer.validate(_Tiny(), dataloaders=_loader())
    assert (tmp_path / "git_sha.txt").read_text() == "d" * 40 + "\n"
