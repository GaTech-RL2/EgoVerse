"""Set child ownership after exec, before importing simulator libraries.

Avoid Python/ctypes preexec callbacks in Ray's multithreaded actor processes.
The Linux parent-death signal and inherited request pipe are preserved.
"""
import ctypes
import faulthandler
import os
import runpy
import signal
import sys


def main():
    parent=int(sys.argv[1]);module=sys.argv[2]
    if sys.platform=='linux':
        library=ctypes.CDLL('libc.so.6',use_errno=True)
        if library.prctl(1,signal.SIGKILL)!=0:raise OSError(ctypes.get_errno(),'Cannot set simulator parent-death signal')
    if os.getppid()!=parent:raise RuntimeError('Simulator owner disappeared before bootstrap')
    print('PEAR_SIMULATOR_EXEC_READY',flush=True)
    if os.environ.get('PEAR_SIMULATOR_STARTUP_DIAGNOSTICS')=='1':
        faulthandler.enable()
        faulthandler.dump_traceback_later(60)
    sys.argv=[module,*sys.argv[3:]]
    runpy.run_module(module,run_name='__main__')


if __name__=='__main__':main()
