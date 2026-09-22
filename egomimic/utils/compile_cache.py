"""Per-job torch.compile cache under Slurm.

Inductor defaults to <tmp>/torchinductor_<user>, so two Slurm jobs of one user
on a node share it. Under Slurm this points TORCHINDUCTOR_CACHE_DIR at
<tmp>/torchinductor_<user>_<SLURM_JOB_ID> (every array task has its own job id);
inductor puts its Triton cache inside it. Outside Slurm torch's default is kept
so local re-runs hit a warm cache. Nothing is deleted at exit: DDP ranks, their
compile workers and other runs in the same allocation may still be using it.

Call it from the job itself, before anything compiles (inductor reads the var
lazily). A value from the user's environment wins; one inherited from a process
that ran this for another job does not (``_MARKER`` tells the two apart).
"""

from __future__ import annotations

import getpass
import os
import tempfile
from typing import MutableMapping, Optional

_MARKER = "_EGOMIMIC_TORCHINDUCTOR_CACHE_DIR"  # the value this module last set


def _user() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):  # no USER/LOGNAME and no passwd entry (containers)
        return "user"


def set_per_job_compile_cache_dir(
    env: MutableMapping[str, str] = os.environ,
) -> Optional[str]:
    current = env.get("TORCHINDUCTOR_CACHE_DIR")
    job = env.get("SLURM_JOB_ID")
    if not job or (current is not None and current != env.get(_MARKER)):
        return current
    current = os.path.join(tempfile.gettempdir(), f"torchinductor_{_user()}_{job}")
    env["TORCHINDUCTOR_CACHE_DIR"] = env[_MARKER] = current
    return current
