"""Linux chroot + dropped UID/capabilities + seccomp; no instruction-only fallback.

The child sees a minimal standard-library runtime, audited source, and its own
scratch directory. It has no simulator object, benchmark assets, credentials,
network sockets, host /proc, or inherited descriptors beyond the JSON pipes.
"""

import contextlib
import ctypes
import errno
import os
import resource
import selectors
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

from .common import encoded, exact, file_hash, strict_json


def require_host():
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("scratch_requires_isolated_Linux_root_container")
    return ctypes.CDLL("libseccomp.so.2", use_errno=True)


def build_runtime(root):
    require_host()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    executable = Path(sys.executable).resolve()
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    destination = root / str(stdlib).lstrip("/")
    shutil.copytree(
        stdlib,
        destination,
        ignore=shutil.ignore_patterns(
            "site-packages",
            "dist-packages",
            "__pycache__",
            "test",
            "tests",
            "ensurepip",
            "idlelib",
            "tkinter",
        ),
    )
    pending, copied = [executable], set()
    # lib-dynload extensions also depend on libraries that Python itself may not.
    pending.extend(stdlib.glob("lib-dynload/*.so"))
    while pending:
        source = Path(pending.pop())
        if str(source) in copied:
            continue
        copied.add(str(source))
        target = root / str(source).lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        target.chmod(0o555)
        output = subprocess.run(
            ["ldd", str(source)], capture_output=True, text=True, check=False
        ).stdout
        for line in output.splitlines():
            parts = line.strip().split()
            candidate = (
                parts[2]
                if len(parts) > 2 and parts[1] == "=>"
                else parts[0]
                if parts
                else ""
            )
            if candidate.startswith("/") and Path(candidate).is_file():
                pending.append(Path(candidate))
    shutil.copyfile(Path(__file__).with_name("scratch_worker.py"), root / "worker.py")
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() or os.access(str(path), os.X_OK) else 0o444)
    root.chmod(0o555)
    return {
        "executable": str(executable),
        "worker_sha256": file_hash(root / "worker.py"),
        "isolation": "chroot; uid/gid 65534; no_new_privs; seccomp deny sockets and namespace/ptrace operations",
    }


class Scratch:
    def __init__(self, runtime, executable, trial_directory, source_view, condition):
        self.seccomp = require_host()
        self.condition, self.executable = condition, executable
        self.last_diagnostic = ""
        self.root = Path(trial_directory).resolve()
        shutil.copytree(runtime, self.root, copy_function=os.link)
        self.root.chmod(0o755)
        shutil.copytree(source_view, self.root / "sources")
        scratch = self.root / "scratch"
        scratch.mkdir(mode=0o700)
        os.chown(str(scratch), 65534, 65534)
        self.root.chmod(0o555)

    def _restrict(self):
        lib = self.seccomp
        lib.seccomp_init.argtypes = [ctypes.c_uint32]
        lib.seccomp_init.restype = ctypes.c_void_p
        lib.seccomp_rule_add.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
        lib.seccomp_load.argtypes = [ctypes.c_void_p]
        lib.seccomp_release.argtypes = [ctypes.c_void_p]
        context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
        if not context:
            os._exit(121)
        for name in (
            "socket",
            "socketpair",
            "connect",
            "bind",
            "listen",
            "accept",
            "accept4",
            "ptrace",
            "mount",
            "umount2",
            "unshare",
            "setns",
            "bpf",
            "userfaultfd",
            "process_vm_readv",
            "process_vm_writev",
            "kill",
            "tkill",
            "tgkill",
            "clone",
            "clone3",
            "fork",
            "vfork",
        ):
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if (
                number >= 0
                and lib.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0)
                != 0
            ):
                os._exit(122)
        os.chroot(str(self.root))
        os.chdir("/scratch")
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
        libc = ctypes.CDLL(None)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            os._exit(123)
        if lib.seccomp_load(context) != 0:
            os._exit(124)
        lib.seccomp_release(context)
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2,) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def execute(self, code, dispatch, *, seconds=15):
        if type(code) is not str or not 1 <= len(code.encode()) <= 16384:
            return {"error": "scratch_code_size"}
        process = subprocess.Popen(
            [self.executable, "-I", "-S", "/worker.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                "PATH": "/nonexistent",
                "LANG": "C.UTF-8",
                "PYTHONHASHSEED": "0",
                # The minimal jail deliberately has no host ld.so.cache. These
                # are paths inside the jail, containing only copied runtime libs.
                "LD_LIBRARY_PATH": "/usr/local/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib64",
            },
            preexec_fn=self._restrict,
            close_fds=True,
            start_new_session=True,
        )
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline, buffer = time.monotonic() + seconds, b""
        try:
            process.stdin.write(
                encoded({"code": code, "condition": self.condition}) + b"\n"
            )
            process.stdin.flush()
            while time.monotonic() < deadline:
                ready = selector.select(min(0.1, max(0, deadline - time.monotonic())))
                if not ready:
                    if process.poll() is not None:
                        return {
                            "error": "scratch_process_exit",
                            "exit_code": process.returncode,
                        }
                    continue
                data = os.read(process.stdout.fileno(), 65536)
                if not data:
                    return {"error": "scratch_process_exit"}
                buffer += data
                if len(buffer) > 1024**2:
                    return {"error": "scratch_output_limit"}
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    message = strict_json(line)
                    if set(message) in ({"result"}, {"error"}):
                        return message
                    exact(message, ("tool", "arguments"))
                    allowed = (
                        {"read", "act"}
                        if self.condition == "F"
                        else {"observe", "step"}
                    )
                    if message["tool"] not in allowed:
                        return {"error": "scratch_tool_not_allowed"}
                    result = dispatch(message["tool"], message["arguments"])
                    process.stdin.write(encoded(result) + b"\n")
                    process.stdin.flush()
            return {"error": "scratch_timeout"}
        except (ValueError, BrokenPipeError, OSError):
            return {"error": "scratch_protocol"}
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            selector.close()
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()
            process.stdout.close()
            self.last_diagnostic = process.stderr.read(65536).decode(errors="replace")
            process.stderr.close()
