"""The OSMO launcher must parse as YAML and its entry script as bash.

Both failures are rejected at SUBMIT time with a 400, so they cost minutes
rather than hours — but they cost a full submit/diagnose cycle each, and both
are detectable locally in milliseconds.

The one that actually happened: a `python -c "` block whose continuation lines
sat at column 1 escaped the YAML literal block, and the spec stopped parsing:

    Workflow spec is not properly formatted: while scanning a simple key
      in "<unicode string>", line 106, column 1:
        from Tsimulation.pushshapes impo ...
    could not find expected ':'

Embedded interpreters belong in an INDENTED heredoc (`python - <<'PY'`), the
way the wandb preflight already does it, so every line stays inside the block.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

yaml = pytest.importorskip("yaml")

REPO = pathlib.Path(__file__).resolve().parents[1]
LAUNCHERS = sorted((REPO / "osmo").glob("pushshapes_*.yaml"))


def _load(path: pathlib.Path):
    """Parse with the `{{jinja}}` placeholders neutralised.

    osmo substitutes them before parsing; bare `{{x}}` is not valid YAML.
    """
    raw = re.sub(r"\{\{\s*([\w.]+)\s*\}\}", r"PLACEHOLDER_\1", path.read_text())
    return yaml.safe_load(raw)


@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_launcher_is_valid_yaml(path):
    spec = _load(path)
    assert spec["workflow"]["tasks"], f"{path.name}: no tasks"


@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_entry_script_is_valid_bash(path, tmp_path):
    """`bash -n` the embedded entry script.

    A syntax error here is only discovered on the node, after the image pull.
    """
    spec = _load(path)
    for task in spec["workflow"]["tasks"]:
        for f in task.get("files") or []:
            contents = f.get("contents")
            if not contents or not str(f.get("path", "")).endswith(".sh"):
                continue
            script = tmp_path / "entry.sh"
            script.write_text(contents)
            out = subprocess.run(["bash", "-n", str(script)],
                                 capture_output=True, text=True)
            assert out.returncode == 0, (
                f"{path.name} {f['path']}: bash syntax error\n{out.stderr}")


def test_control_mode_launcher_installs_the_simulator_deps():
    """pymunk/gymnasium/pygame/shapely are absent from EgoVerse's uv.lock.

    Tsimulation imports all four at module scope, so without an explicit
    install SimRolloutEval cannot construct an env and the evaluator dies at
    its first rollout — in phase 2, after training has begun.
    """
    path = REPO / "osmo/pushshapes_control_modes_l40s.yaml"
    entry = _load(path)["workflow"]["tasks"][0]["files"][0]["contents"]
    for dep in ("pymunk", "gymnasium", "pygame", "shapely"):
        assert dep in entry, f"{dep} is not installed by the launcher"
    # 6.9.0 fails on reset; the sim uses the 7.x Space.on_collision API.
    assert 'pymunk==7.3.0' in entry, "pymunk must be pinned to 7.x"
    # And the install must be verified, not assumed.
    assert "from Tsimulation.pushshapes import PushShapesEnv" in entry


def test_simulator_deps_are_still_missing_from_the_lock():
    """Guards the workaround.

    If pymunk et al. are ever added to pyproject/uv.lock properly, the
    launcher's `uv pip install` becomes redundant and this test should be the
    thing that says so — rather than the workaround quietly outliving its
    reason.
    """
    lock = (REPO / "uv.lock").read_text()
    still_missing = [d for d in ("pymunk", "gymnasium", "pygame", "shapely")
                     if f'name = "{d}"' not in lock]
    assert still_missing == ["pymunk", "gymnasium", "pygame", "shapely"], (
        f"uv.lock now provides {set(['pymunk','gymnasium','pygame','shapely']) - set(still_missing)}; "
        "drop them from the launcher's uv pip install")
