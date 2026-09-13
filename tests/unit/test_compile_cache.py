"""Under Slurm the torch.compile cache is per job so two jobs on one node never
share <tmp>/torchinductor_<user> (Phase 4)."""

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import egomimic
from egomimic.utils.compile_cache import set_per_job_compile_cache_dir

KEY = "TORCHINDUCTOR_CACHE_DIR"
ROOT = Path(egomimic.__file__).parents[1]


def test_keys_on_slurm_job_id_under_tempdir(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # as TMPDIR would
    env = {"SLURM_JOB_ID": "4242"}
    out = set_per_job_compile_cache_dir(env)
    assert os.path.dirname(out) == str(tmp_path)
    assert os.path.basename(out).startswith("torchinductor_")
    assert out.endswith("_4242")
    assert env[KEY] == out
    assert "TRITON_CACHE_DIR" not in env  # inductor derives it from KEY


def test_outside_slurm_keeps_torch_default():
    env = {}
    assert set_per_job_compile_cache_dir(env) is None
    assert env == {}


def test_explicit_value_is_respected():
    env = {KEY: "/scratch/ind", "SLURM_JOB_ID": "1"}
    assert set_per_job_compile_cache_dir(env) == "/scratch/ind"
    assert env[KEY] == "/scratch/ind"


_LAUNCHER = """
import os, subprocess, sys
from egomimic.utils.compile_cache import set_per_job_compile_cache_dir as s
print(s())
worker = "from egomimic.utils.compile_cache import set_per_job_compile_cache_dir as s; print(s())"
env = dict(os.environ, SLURM_JOB_ID="200")
sys.stdout.flush()
subprocess.run([sys.executable, "-c", worker], env=env, check=True)
"""


def _launch(user_value=None):
    """A process in Slurm job 100 sets the cache dir, then starts job 200 with
    its environment (what sbatch does). Returns (launcher dir, worker dir)."""
    env = {k: v for k, v in os.environ.items() if not k.endswith(KEY)}
    env["SLURM_JOB_ID"] = "100"
    if user_value is not None:
        env[KEY] = user_value
    res = subprocess.run(
        [sys.executable, "-c", _LAUNCHER],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=300,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout.split()


def test_worker_does_not_inherit_the_launchers_job_dir():
    launcher, worker = _launch()
    assert launcher.endswith("_100")
    assert worker.endswith("_200")


def test_worker_keeps_a_user_value_from_the_launchers_shell():
    assert _launch("/scratch/ind") == ["/scratch/ind", "/scratch/ind"]


def test_trainhydra_sets_it_in_main_not_at_import():
    """A `-m` launcher imports trainHydra but never runs main(); setting it at
    import would key every job of a sweep on the launcher."""
    src = (ROOT / "egomimic" / "trainHydra.py").read_text()
    tree = ast.parse(src)

    def calls(nodes):
        return any(
            isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "set_per_job_compile_cache_dir"
            for node in nodes
            for n in ast.walk(node)
        )

    top = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    assert not calls(top)
    assert calls(main.body)
