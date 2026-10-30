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

from egomimic.utils import gpu_orphans
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


# The wait channel of a writer blocked on a full pipe: split out of pipe_wait in
# Linux 5.5, so a 5.4 kernel (sky2) still reports the old name.
_PIPE_BLOCKED = {"pipe_write", "pipe_wait"}


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
        wchan = ""
        while time.time() < deadline:
            wchan = open(f"/proc/{worker}/wchan").read()
            if wchan in _PIPE_BLOCKED:
                break
            time.sleep(0.2)
        else:
            os.kill(worker, 9)  # an unguarded worker would outlive the test
            pytest.skip(f"this kernel reports the blocked worker as {wchan!r}")
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


def test_orphan_guard_runs_before_a_caller_init(monkeypatch):
    calls = []
    # never the real prctl: it would arm PDEATHSIG on pytest itself
    monkeypatch.setattr(
        gpu_orphans, "die_with_parent", lambda i=0: calls.append(("guard", i))
    )
    params = orphan_guarded(
        {"num_workers": 2, "worker_init_fn": lambda i: calls.append(("user", i))}
    )
    params["worker_init_fn"](3)
    assert calls == [("guard", 3), ("user", 3)]


class _Exited(Exception):
    pass


@pytest.mark.parametrize(
    "fork_parent, ppid, exits",
    [
        (1, 1, False),  # the rank is PID 1 (container entrypoint)
        (4242, 4242, False),
        (4242, 1, True),  # the rank died during the fork
    ],
)
def test_die_with_parent_checks_the_pid_that_forked_it(
    monkeypatch, fork_parent, ppid, exits
):
    class _Libc:
        def prctl(self, *a):
            return 0

    def _exit(code):
        raise _Exited(code)

    monkeypatch.setattr(gpu_orphans.ctypes, "CDLL", lambda *a, **k: _Libc())
    monkeypatch.setattr(gpu_orphans, "_fork_parent", fork_parent)
    monkeypatch.setattr(gpu_orphans.os, "getppid", lambda: ppid)
    monkeypatch.setattr(gpu_orphans.os, "_exit", _exit)
    monkeypatch.setattr(gpu_orphans.os, "write", lambda *a: None)
    if exits:
        with pytest.raises(_Exited):
            die_with_parent(0)
    else:
        die_with_parent(0)


@pytest.mark.skipif(sys.platform != "linux", reason="os.fork")
def test_a_forked_child_records_its_forking_parent():
    r, w = os.pipe()
    parent = os.getpid()
    pid = os.fork()
    if pid == 0:
        os.write(w, str(gpu_orphans._fork_parent).encode())
        os._exit(0)
    os.close(w)
    os.waitpid(pid, 0)
    assert os.read(r, 64).decode() == str(parent)


def test_orphan_guard_is_a_noop_without_workers():
    assert orphan_guarded({"num_workers": 0}) == {"num_workers": 0}


def test_orphan_guard_installs_the_default():
    assert orphan_guarded({"num_workers": 4})["worker_init_fn"] is die_with_parent


def _fake_proc(root, pid: int, *, ppid: int, fds: list[str], job="7") -> None:
    d = root / str(pid)
    (d / "fd").mkdir(parents=True)
    (d / "status").write_text(f"Name:\tpt_data_worker\nPPid:\t{ppid}\n")
    env = [b"HOME=/x"] + ([f"SLURM_JOB_ID={job}".encode()] if job else [])
    (d / "environ").write_bytes(b"\0".join(env) + b"\0")
    for i, target in enumerate(fds):
        os.symlink(target, d / "fd" / str(i))


def _ENDED(job):
    return True


def test_find_orphans_picks_only_ours_orphaned_and_on_a_gpu(tmp_path):
    uid = os.getuid()
    gpu = ["/dev/nvidia0", "/dev/nvidiactl"]
    _fake_proc(tmp_path, 100, ppid=1, fds=gpu)  # the leak
    _fake_proc(tmp_path, 101, ppid=42, fds=gpu)  # live job's worker
    _fake_proc(tmp_path, 102, ppid=1, fds=["/dev/null"])  # no GPU
    assert find_orphans(str(tmp_path), uid=uid, job_ended=_ENDED) == [100]


def test_find_orphans_spares_live_jobs_and_detached_runs(tmp_path, monkeypatch):
    """A nohup run whose shell exited is PPID 1 and on a GPU, but alive."""
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    uid = os.getuid()
    gpu = ["/dev/nvidia0"]
    _fake_proc(tmp_path, 400, ppid=1, fds=gpu, job="7")  # job 7 is over
    _fake_proc(tmp_path, 401, ppid=1, fds=gpu, job="8")  # job 8 still runs
    _fake_proc(tmp_path, 402, ppid=1, fds=gpu, job="9")  # this job's nohup run
    _fake_proc(tmp_path, 403, ppid=1, fds=gpu, job=None)  # not from a job
    asked = []

    def ended(job):
        asked.append(job)
        return job == "7"

    assert find_orphans(str(tmp_path), uid=uid, job_ended=ended) == [400]
    assert sorted(asked) == ["7", "8"]


@pytest.mark.parametrize(
    "rc, out, err, ended",
    [
        (0, "RUNNING\n", "", False),
        (0, "COMPLETING\n", "", False),
        (0, "CANCELLED\n", "", True),
        (0, "", "", True),
        (1, "", "slurm_load_jobs error: Invalid job id specified", True),
        (1, "", "Socket timed out on send/recv operation", False),
    ],
)
def test_job_ended_reads_squeue(monkeypatch, rc, out, err, ended):
    import types

    monkeypatch.setattr(
        gpu_orphans.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=rc, stdout=out, stderr=err),
    )
    assert gpu_orphans._job_ended("7") is ended


def test_job_ended_without_squeue_is_false(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("squeue")

    monkeypatch.setattr(gpu_orphans.subprocess, "run", missing)
    assert gpu_orphans._job_ended("7") is False


def test_find_orphans_skips_other_users(tmp_path):
    uid = os.getuid()
    _fake_proc(tmp_path, 200, ppid=1, fds=["/dev/nvidia0"])
    assert find_orphans(str(tmp_path), uid=uid + 1, job_ended=_ENDED) == []


def test_find_orphans_tolerates_a_process_exiting_mid_scan(tmp_path):
    uid = os.getuid()
    (tmp_path / "300").mkdir()  # no status/fd: raced away
    _fake_proc(tmp_path, 301, ppid=1, fds=["/dev/nvidia0"])
    assert find_orphans(str(tmp_path), uid=uid, job_ended=_ENDED) == [301]


def test_reap_is_off_outside_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert reap_orphans() == []


def test_reap_runs_once_per_node(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_LOCALID", "3")
    assert reap_orphans() == []


@pytest.mark.parametrize(
    "value, on",
    [("0", False), ("false", False), ("1", True), ("true", True), ("yes", True)],
)
def test_reap_honours_the_kill_switch(monkeypatch, value, on):
    """Any truthy spelling keeps it on. Never the real /proc: a regression here
    would SIGKILL the tester's own orphans."""
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("EGOMIMIC_REAP_GPU_ORPHANS", value)
    monkeypatch.setattr(gpu_orphans, "find_orphans", lambda: [4242])
    assert reap_orphans(dry_run=True) == ([4242] if on else [])


def test_reap_rejects_a_misspelled_switch(monkeypatch):
    import argparse

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("EGOMIMIC_REAP_GPU_ORPHANS", "of")
    monkeypatch.setattr(gpu_orphans, "find_orphans", lambda: [4242])
    with pytest.raises(argparse.ArgumentTypeError):
        reap_orphans(dry_run=True)


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


def test_rank_pdeathsig_leaves_a_nohup_run_alone(monkeypatch):
    """nohup inside salloc means "outlive this shell"; arming would kill it."""
    import signal

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        assert arm_rank_pdeathsig() is False
    finally:
        signal.signal(signal.SIGHUP, previous)
