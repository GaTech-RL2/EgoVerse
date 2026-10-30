"""Reclaim GPUs pinned by dataloader workers that outlived a cancelled job.

A worker forked after CUDA init holds its rank's ``/dev/nvidia*`` fds, so the
rank's whole context stays resident after the rank is gone -- ``nvidia-smi`` then
reports the memory against a PID that is no longer in ``/proc``. The worker
cannot exit on its own (it blocks in ``pipe_write`` on a result pipe nobody
drains), and ``ProctrackType=proctrack/linuxproc`` walks the PPID tree, so
Slurm never sweeps it once it is reparented to init.

``orphan_guarded`` / ``die_with_parent`` stop new leaks. This is the other half: a job
clears what an earlier job left on its node, before it touches CUDA itself.

Nothing is imported from torch here -- this has to run before CUDA init.
"""

import ctypes
import logging
import os
import signal
import subprocess
from typing import Callable

from egomimic.utils.type_utils import str2bool

logger = logging.getLogger(__name__)

_REAP_ENV = "EGOMIMIC_REAP_GPU_ORPHANS"
_PR_SET_PDEATHSIG = 1

# Pid of the process that os.fork()ed this one, set in the child. getppid() == 1
# cannot tell a dead parent from a rank that is itself PID 1 (a container
# entrypoint), whose every worker is legitimately parented to 1.
_fork_parent: int | None = None
_forking_pid: int | None = None


def _before_fork() -> None:
    global _forking_pid
    _forking_pid = os.getpid()


def _after_fork_in_child() -> None:
    global _fork_parent
    _fork_parent = _forking_pid


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_before_fork, after_in_child=_after_fork_in_child)


def die_with_parent(_worker_id: int = 0) -> None:
    """Ask the kernel to SIGKILL this dataloader worker when its rank dies.

    PyTorch's own parent check only runs between ``index_queue.get()`` timeouts,
    which a worker blocked writing to a full result pipe never reaches. The
    kernel does not care what the worker is blocked on.
    """
    parent = _fork_parent if _fork_parent is not None else os.getppid()
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            _PR_SET_PDEATHSIG, signal.SIGKILL
        )
    except (OSError, AttributeError):  # non-Linux, or no libc
        logger.debug("PR_SET_PDEATHSIG unavailable; orphaned workers may leak GPUs")
        return
    # pdeathsig does not fire retroactively if the rank died during the fork.
    if os.getppid() != parent:
        # os._exit skips logging's flush; say why before a rank exits 1 silently.
        os.write(
            2,
            f"[gpu_orphans] pid {os.getpid()}: parent gone before "
            "PR_SET_PDEATHSIG was armed; exiting\n".encode(),
        )
        os._exit(1)


def orphan_guarded(params: dict) -> dict:
    """``params`` with ``die_with_parent`` run before any caller ``worker_init_fn``."""
    params = dict(params)
    if params.get("num_workers", 0) == 0:
        return params
    user_init = params.get("worker_init_fn")
    if user_init is None:
        params["worker_init_fn"] = die_with_parent
    else:

        def _chained(worker_id: int, _user_init=user_init) -> None:
            die_with_parent(worker_id)
            _user_init(worker_id)

        params["worker_init_fn"] = _chained
    return params


def _ppid(proc_root: str, pid: str) -> int | None:
    try:
        with open(f"{proc_root}/{pid}/status") as f:
            for line in f:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def _holds_gpu(proc_root: str, pid: str) -> bool:
    fd_dir = f"{proc_root}/{pid}/fd"
    try:
        fds = os.listdir(fd_dir)
    except OSError:  # gone, or not ours
        return False
    for fd in fds:
        try:
            if os.readlink(f"{fd_dir}/{fd}").startswith("/dev/nvidia"):
                return True
        except OSError:
            continue
    return False


def _slurm_job_id(proc_root: str, pid: str) -> str | None:
    try:
        with open(f"{proc_root}/{pid}/environ", "rb") as f:
            for entry in f.read().split(b"\0"):
                if entry.startswith(b"SLURM_JOB_ID="):
                    return entry.split(b"=", 1)[1].decode()
    except OSError:
        pass
    return None


# squeue %T states of a job that has left the node for good.
_ENDED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


def _job_ended(job_id: str) -> bool:
    """True only when Slurm says the job is over; any doubt keeps it alive."""
    try:
        r = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if r.returncode != 0:
        return "invalid job id" in r.stderr.lower()
    state = r.stdout.strip()
    return not state or state in _ENDED_STATES


def arm_rank_pdeathsig() -> bool:
    """Have the kernel kill this process if its launcher dies first.

    Workers are not the only thing that leaks. A rank wedged in
    ``torch.cuda.synchronize()`` inside inductor's cudagraph_trees does not
    answer SIGTERM; Slurm hits ``UnkillableStepTimeout``, gives up, and the job
    leaves the queue with its ranks still spinning on every GPU of the node.

    Only armed under Slurm, where the parent is slurmstepd or the launcher that
    spawned this rank and never legitimately exits first. Off elsewhere, and off
    for a ``nohup`` run (SIGHUP ignored): inside a ``salloc`` shell that is a run
    meant to outlive the shell, which PR_SET_PDEATHSIG would kill.
    """
    if "SLURM_JOB_ID" not in os.environ:
        return False
    if signal.getsignal(signal.SIGHUP) is signal.SIG_IGN:
        logger.info("SIGHUP is ignored (nohup); not tying this rank to its parent")
        return False
    die_with_parent()
    return True


def find_orphans(
    proc_root: str = "/proc",
    uid: int | None = None,
    job_ended: Callable[[str], bool] = _job_ended,
) -> list[int]:
    """PIDs that are ours, reparented to init, holding a GPU device open, and
    left by a Slurm job that has ended.

    PPID 1 alone is not enough: a ``nohup`` run (or any detached GPU process)
    whose shell exited is reparented to init while still doing its job. So the
    process must also carry another job's ``SLURM_JOB_ID`` and Slurm must say
    that job is over; one with no job id, or from this job, is left alone.
    """
    uid = os.getuid() if uid is None else uid
    mine = {os.getpid(), os.getppid()}
    this_job = os.environ.get("SLURM_JOB_ID")
    ended: dict[str, bool] = {}
    orphans = []
    for pid in os.listdir(proc_root):
        if not pid.isdigit() or int(pid) in mine:
            continue
        try:
            if os.stat(f"{proc_root}/{pid}").st_uid != uid:
                continue
        except OSError:
            continue
        if _ppid(proc_root, pid) != 1:
            continue
        if not _holds_gpu(proc_root, pid):
            continue
        job = _slurm_job_id(proc_root, pid)
        if job is None or job == this_job:
            continue
        if job not in ended:
            ended[job] = job_ended(job)
        if ended[job]:
            orphans.append(int(pid))
    return sorted(orphans)


def reap_orphans(dry_run: bool = False) -> list[int]:
    """Kill this node's leaked GPU holders. Returns the PIDs signalled.

    Off outside Slurm: only there can a GPU holder be traced to a job that
    has ended (``find_orphans``).
    """
    if not str2bool(os.environ.get(_REAP_ENV, "1")):
        return []
    if "SLURM_JOB_ID" not in os.environ:
        return []
    if os.environ.get("SLURM_LOCALID", "0") != "0":  # once per node
        return []

    orphans = find_orphans()
    if not orphans:
        return []
    logger.warning(
        "Reaping %d orphaned GPU holder(s) left by an earlier job on %s: %s "
        "(set %s=0 to disable)",
        len(orphans),
        os.uname().nodename,
        orphans,
        _REAP_ENV,
    )
    if dry_run:
        return orphans
    for pid in orphans:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as e:
            logger.warning("could not kill orphan %d: %s", pid, e)
    return orphans
