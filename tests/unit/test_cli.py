import subprocess
import sys

import zarr
from fixtures.synthetic_episodes import write_episode

from egomimic import cli
from egomimic.rldb.zarr import schema


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "egomimic.cli", *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_validate_directory_of_episodes_exit_codes(tmp_path) -> None:
    write_episode(tmp_path, "aria", seed=0)
    bad = write_episode(tmp_path, "aria", seed=1)
    ok = _cli("validate", str(tmp_path))
    assert ok.returncode == 0, ok.stderr
    assert "OK   " in ok.stdout and "aria_00" in ok.stdout
    g = zarr.open_group(str(bad), mode="r+")
    g.attrs["fps"] = 25
    res = _cli("validate", str(tmp_path))
    assert res.returncode == 1
    assert "FAIL aria_01" in res.stdout and "fps=25" in res.stdout


def test_schema_prints_markdown() -> None:
    res = _cli("schema")
    assert res.returncode == 0
    assert "| `format_version` |" in res.stdout


def test_validate_reports_crashing_episode_and_continues(
    tmp_path, monkeypatch, capsys
) -> None:
    for seed in range(3):
        write_episode(tmp_path, "aria", seed=seed)
    real = schema.validate_episode

    def flaky(path):
        if path.name == "aria_01.zarr":
            raise IndexError("boom")
        return real(path)

    monkeypatch.setattr(schema, "validate_episode", flaky)
    assert cli._validate([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "FAIL aria_01: validator crashed: IndexError: boom" in out
    assert "OK   aria_00" in out and "OK   aria_02" in out
