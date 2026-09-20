"""Reclaim GPUs pinned by dataloader workers that outlived a cancelled job.

A worker forked after CUDA init holds its rank's ``/dev/nvidia*`` fds, so the
rank's whole context stays resident after the rank is gone -- ``nvidia-smi`` then
reports the memory against a PID that is no longer in ``/proc``. The worker
cannot exit on its own (it blocks in ``pipe_write`` on a result pipe nobody
drains), and ``ProctrackType=proctrack/linuxproc`` walks the PPID tree, so
Slurm never sweeps it once it is reparented to init.

``pl_data_utils._orphan_guarded`` stops new leaks. This is the other half: a job
clears what an earlier job left on its node, before it touches CUDA itself.

Nothing is imported from torch here -- this has to run before CUDA init.
"""

import ctypes
import logging
import os
import signal

logger = logging.getLogger(__name__)

_DISABLE_ENV = "EGOMIMIC_REAP_GPU_ORPHANS"
_PR_SET_PDEATHSIG = 1


def die_with_parent(_worker_id: int = 0) -> None:
    """Ask the kernel to SIGKILL this dataloader worker when its rank dies.

    PyTorch's own parent check only runs between ``index_queue.get()`` timeouts,
    which a worker blocked writing to a full result pipe never reaches. The
    kernel does not care what the worker is blocked on.
    """
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            _PR_SET_PDEATHSIG, signal.SIGKILL
        )
    except (OSError, AttributeError):  # non-Linux, or no libc
        logger.debug("PR_SET_PDEATHSIG unavailable; orphaned workers may leak GPUs")
        return
    # pdeathsig does not fire retroactively if the rank died during the fork.
    if os.getppid() == 1:
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


def arm_rank_pdeathsig() -> bool:
    """Have the kernel kill this process if its launcher dies first.

    Workers are not the only thing that leaks. A rank wedged in
    ``torch.cuda.synchronize()`` inside inductor's cudagraph_trees does not
    answer SIGTERM; Slurm hits ``UnkillableStepTimeout``, gives up, and the job
    leaves the queue with its ranks still spinning on every GPU of the node.

    Only armed under Slurm, where the parent is slurmstepd or the launcher that
    spawned this rank and never legitimately exits first. Off elsewhere, so a
    detached interactive run is not shot when its shell goes away.
    """
    if "SLURM_JOB_ID" not in os.environ:
        return False
    die_with_parent()
    return True


def find_orphans(proc_root: str = "/proc", uid: int | None = None) -> list[int]:
    """PIDs that are ours, reparented to init, and holding a GPU device open.

    A live job's workers always have a live parent, so the PPID-1 test alone
    keeps this off anything still running -- ours or anyone else's.
    """
    uid = os.getuid() if uid is None else uid
    mine = {os.getpid(), os.getppid()}
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
        if _holds_gpu(proc_root, pid):
            orphans.append(int(pid))
    return sorted(orphans)


def reap_orphans(dry_run: bool = False) -> list[int]:
    """Kill this node's leaked GPU holders. Returns the PIDs signalled.

    Off outside Slurm: only there do we know every process of ours on the node
    belongs to a job, so that a PPID-1 GPU holder can only be wreckage.
    """
    if os.environ.get(_DISABLE_ENV, "1") != "1":
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
        _DISABLE_ENV,
    )
    if dry_run:
        return orphans
    for pid in orphans:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as e:
            logger.warning("could not kill orphan %d: %s", pid, e)
    return orphans
