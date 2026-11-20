"""Run an unchanged RTX service with its end callback on the Kit owner thread.

The native callback steps and renders the last command before dumping the live
verdict. Isaac can block when that callback runs on a new daemon thread. The
collect hook is invoked only after the client has stopped sending commands.
This changes callback scheduling, never its steps, success term or verdict.
"""
import runpy
import sys
import threading


def owner_thread_end(episode, *_):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('The native end signal must run on the simulator owner thread')
    if episode._on_end is None:
        episode._end_pending = True
    else:
        episode._on_end()


def main():
    from pathlib import Path
    service = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(service.parent))
    import rcb_limited
    rcb_limited.Episode._usr1 = owner_thread_end
    import os
    if os.environ.get('PEAR_NATIVE_CONTROL_INTERFACE'):
        from efe_pear_eef import enabled
        enabled(os.environ['PEAR_NATIVE_CONTROL_INTERFACE'])
        from efe_pear_eef_service import install
        install(rcb_limited)
    sys.argv = [str(service), *sys.argv[2:]]
    runpy.run_path(str(service), run_name='__main__')


if __name__ == '__main__':
    main()
