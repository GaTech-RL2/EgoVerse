"""openpi's transformers_replace overlay: apply is idempotent and gated on the
transformers version, check detects a missing or modified file, and writes never
go through uv's hardlinks or symlinks."""

import os
import stat
import warnings
from pathlib import Path

import pytest

from egomimic import openpi_patch as op

GEMMA = "models/gemma/modeling_gemma.py"
FILES = {
    "models/siglip/check.py": 'ok = transformers.__version__ == "4.53.2"\n',
    GEMMA: "# patched gemma\n",
}


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(op, "installed_transformers_version", lambda: "4.53.2")
    src = tmp_path / "transformers_replace"
    for rel, text in FILES.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text)
    dst = tmp_path / "site" / "transformers"
    dst.mkdir(parents=True)
    return src, dst


def _link_from_cache(tmp_path, dst, text, link=os.link) -> Path:
    cache = tmp_path / "uv-cache" / "modeling_gemma.py"
    cache.parent.mkdir()
    cache.write_text(text)
    target = dst / GEMMA
    target.parent.mkdir(parents=True)
    link(cache, target)
    return cache


def test_stale_then_apply_then_clean(dirs):
    src, dst = dirs
    assert sorted(map(str, op.status(src, dst)[0])) == sorted(FILES)
    assert sorted(map(str, op.apply(src, dst))) == sorted(FILES)
    assert op.status(src, dst) == ([], [])
    assert op.apply(src, dst) == []  # idempotent
    (dst / GEMMA).write_text("# stock gemma\n")
    assert [str(p) for p in op.status(src, dst)[0]] == [GEMMA]
    assert [str(p) for p in op.apply(src, dst)] == [GEMMA]
    op.check(src, dst)


def test_required_version_comes_from_the_overlay_marker(dirs):
    src, _ = dirs
    assert op.required_transformers(src) == "4.53.2"
    (src / "models/siglip/check.py").write_text("import transformers\n")
    with pytest.raises(RuntimeError, match="no transformers version check"):
        op.required_transformers(src)


@pytest.mark.parametrize("link", [os.link, os.symlink])
def test_apply_does_not_write_through_links(tmp_path, dirs, link):
    src, dst = dirs
    cache = _link_from_cache(tmp_path, dst, "# stock gemma\n", link)
    op.apply(src, dst)
    target = dst / GEMMA
    assert target.read_text() == "# patched gemma\n"
    assert not target.is_symlink() and os.stat(target).st_nlink == 1
    assert cache.read_text() == "# stock gemma\n"


def test_apply_replaces_dangling_symlink_without_following_it(tmp_path, dirs):
    src, dst = dirs
    elsewhere = tmp_path / "elsewhere.py"
    (dst / GEMMA).parent.mkdir(parents=True)
    (dst / GEMMA).symlink_to(elsewhere)
    op.apply(src, dst)
    assert not elsewhere.exists()
    assert not (dst / GEMMA).is_symlink()
    assert (dst / GEMMA).read_text() == "# patched gemma\n"


def test_apply_keeps_target_mode(dirs):
    src, dst = dirs
    target = dst / GEMMA
    target.parent.mkdir(parents=True)
    target.write_text("# stock gemma\n")
    target.chmod(0o644)
    old = os.umask(0o077)
    try:
        op.apply(src, dst)
    finally:
        os.umask(old)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert not [p for p in target.parent.iterdir() if p.name.startswith(".")]


@pytest.mark.parametrize("link", [os.link, os.symlink])
def test_linked_target_that_already_matches_warns_about_the_cache(
    tmp_path, dirs, monkeypatch, capsys, link
):
    """The old `cp -r` wrote the patch through uv's link, so the cache itself
    holds the patched bytes; rewriting the venv file cannot fix that."""
    src, dst = dirs
    cache = _link_from_cache(tmp_path, dst, "# patched gemma\n", link)
    with pytest.warns(UserWarning, match="uv cache clean transformers"):
        assert [str(p) for p in op.apply(src, dst)] == ["models/siglip/check.py"]
    monkeypatch.setattr(op, "source_dir", lambda: src)
    monkeypatch.setattr(op, "target_dir", lambda: dst)
    assert op.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "WARNING:" in out and "uv cache clean transformers" in out
    cache.unlink()  # what `uv cache clean transformers` does
    op.apply(src, dst)  # rewrites a now-dangling symlink; a hardlink is left alone
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        op.check(src, dst)


@pytest.mark.parametrize("make", [lambda src: None, lambda src: src.mkdir()])
def test_missing_or_empty_overlay_is_an_error(tmp_path, monkeypatch, make):
    monkeypatch.setattr(op, "installed_transformers_version", lambda: "4.53.2")
    src, dst = tmp_path / "transformers_replace", tmp_path / "transformers"
    make(src)
    dst.mkdir()
    for fn in (op.check, op.apply):
        with pytest.raises(RuntimeError, match="missing or empty"):
            fn(src, dst)


def test_apply_refuses_other_transformers(dirs, monkeypatch):
    src, dst = dirs
    monkeypatch.setattr(op, "installed_transformers_version", lambda: "4.57.3")
    with pytest.raises(RuntimeError, match="transformers==4.53.2, found 4.57.3"):
        op.apply(src, dst)
    assert not any(dst.iterdir())
    with pytest.raises(RuntimeError, match="4.53.2"):
        op.check(src, dst)


def test_check_names_the_fix_command(dirs):
    src, dst = dirs
    with pytest.raises(RuntimeError, match=r"python -m egomimic\.openpi_patch apply"):
        op.check(src, dst)


def test_cli_exit_codes(dirs, monkeypatch, capsys):
    src, dst = dirs
    monkeypatch.setattr(op, "source_dir", lambda: src)
    monkeypatch.setattr(op, "target_dir", lambda: dst)
    assert op.main(["check"]) == 1
    assert "openpi_patch apply" in capsys.readouterr().out
    assert op.main(["apply"]) == 0
    assert "2 file(s) updated" in capsys.readouterr().out
    assert op.main(["check"]) == 0
    assert "OK" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["apply", "check"])
def test_cli_without_openpi_prints_one_line(monkeypatch, capsys, command):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert op.main([command]) == 1
    out = capsys.readouterr().out
    assert out == f"openpi is not installed; run: {op.INSTALL_CMD}\n"
