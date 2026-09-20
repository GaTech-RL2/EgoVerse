"""Dataloader workers must die with their rank.

Workers are forked after CUDA init, so they inherit the rank's ``/dev/nvidia*``
fds. One survivor keeps every GPU context on the node alive, and
``proctrack/linuxproc`` will not sweep it because it has been reparented to init.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

from egomimic.utils.gpu_orphans import (
    arm_rank_pdeathsig,
    die_with_parent,
    find_orphans,
    orphan_guarded,
    reap_orphans,
)

# A worker that inherits both ends of its result pipe never gets an EPIPE when
# the rank dies; it blocks in pipe_write, so PyTorch's own ppid check -- which
# only runs between index_queue timeouts -- is never reached.
RANK = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, {root!r})
    from egomimic.utils.gpu_orphans import die_with_parent
    guard = sys.argv[1] == "guard"
    r, w = os.pipe()
    if os.fork() == 0:
        if guard:
            die_with_parent(0)
        with open(sys.argv[2], "w") as f:
            f.write(str(os.getpid()))
        while True:
            os.write(w, b"x" * 65536)
    time.sleep(600)
    """
)


def _alive(pid: int) -> bool:
    return os.path.isdir(f"/proc/{pid}")


def _pid_written(pidfile) -> bool:
    return pidfile.exists() and pidfile.read_text().strip().isdigit()


def _run_rank(tmp_path, mode: str) -> int:
    """Start a rank, SIGKILL it once its worker is wedged, return the worker pid."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pidfile = tmp_path / f"pid_{mode}"
    rank = subprocess.Popen(
        [sys.executable, "-c", RANK.format(root=root), mode, str(pidfile)]
    )
    try:
        deadline = time.time() + 180  # the rank imports torch first
        while time.time() < deadline and not _pid_written(pidfile):
            time.sleep(0.2)
        assert _pid_written(pidfile), "worker never started"
        worker = int(pidfile.read_text())
        deadline = time.time() + 30
        while time.time() < deadline:
            if open(f"/proc/{worker}/wchan").read() == "pipe_write":
                break
            time.sleep(0.2)
        else:
            pytest.fail("worker never blocked in pipe_write")
    finally:
        rank.kill()
        rank.wait()
    return worker


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux-only")
def test_unguarded_worker_outlives_its_rank(tmp_path):
    worker = _run_rank(tmp_path, "noguard")
    time.sleep(3)
    try:
        assert _alive(worker), "baseline no longer reproduces the leak"
    finally:
        os.kill(worker, 9)


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux-only")
def test_guarded_worker_dies_with_its_rank(tmp_path):
    worker = _run_rank(tmp_path, "guard")
    deadline = time.time() + 15
    while time.time() < deadline and _alive(worker):
        time.sleep(0.2)
    if _alive(worker):
        os.kill(worker, 9)
        pytest.fail("worker survived its rank despite PR_SET_PDEATHSIG")


def test_orphan_guard_runs_before_a_caller_init():
    calls = []
    params = orphan_guarded(
        {"num_workers": 2, "worker_init_fn": lambda i: calls.append(("user", i))}
    )
    params["worker_init_fn"](3)
    assert calls == [("user", 3)]


def test_orphan_guard_is_a_noop_without_workers():
    assert orphan_guarded({"num_workers": 0}) == {"num_workers": 0}


def test_orphan_guard_installs_the_default():
    assert orphan_guarded({"num_workers": 4})["worker_init_fn"] is die_with_parent


def _fake_proc(root, pid: int, *, ppid: int, fds: list[str]) -> None:
    d = root / str(pid)
    (d / "fd").mkdir(parents=True)
    (d / "status").write_text(f"Name:\tpt_data_worker\nPPid:\t{ppid}\n")
    for i, target in enumerate(fds):
        os.symlink(target, d / "fd" / str(i))


def test_find_orphans_picks_only_ours_orphaned_and_on_a_gpu(tmp_path):
    uid = os.getuid()
    gpu = ["/dev/nvidia0", "/dev/nvidiactl"]
    _fake_proc(tmp_path, 100, ppid=1, fds=gpu)  # the leak
    _fake_proc(tmp_path, 101, ppid=42, fds=gpu)  # live job's worker
    _fake_proc(tmp_path, 102, ppid=1, fds=["/dev/null"])  # no GPU
    assert find_orphans(str(tmp_path), uid=uid) == [100]


def test_find_orphans_skips_other_users(tmp_path):
    uid = os.getuid()
    _fake_proc(tmp_path, 200, ppid=1, fds=["/dev/nvidia0"])
    assert find_orphans(str(tmp_path), uid=uid + 1) == []


def test_find_orphans_tolerates_a_process_exiting_mid_scan(tmp_path):
    uid = os.getuid()
    (tmp_path / "300").mkdir()  # no status/fd: raced away
    _fake_proc(tmp_path, 301, ppid=1, fds=["/dev/nvidia0"])
    assert find_orphans(str(tmp_path), uid=uid) == [301]


def test_reap_is_off_outside_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert reap_orphans() == []


def test_reap_runs_once_per_node(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_LOCALID", "3")
    assert reap_orphans() == []


def test_reap_honours_the_kill_switch(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("EGOMIMIC_REAP_GPU_ORPHANS", "0")
    assert reap_orphans() == []


def test_rank_pdeathsig_is_off_outside_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert arm_rank_pdeathsig() is False


def test_rank_pdeathsig_arms_under_slurm(monkeypatch):
    import ctypes

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    try:
        assert arm_rank_pdeathsig() is True
    finally:
        # do not leave pytest itself armed
        ctypes.CDLL("libc.so.6").prctl(1, 0)
