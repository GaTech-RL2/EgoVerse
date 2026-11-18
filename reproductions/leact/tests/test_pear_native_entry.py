import threading
from types import SimpleNamespace

from reproductions.leact.cluster.efe_pear_native_entry import owner_thread_end


def test_collect_callback_runs_on_the_simulator_owner_thread():
    observed = []
    native = SimpleNamespace(_on_end=lambda: observed.append(threading.get_ident()))
    owner_thread_end(native)
    assert observed == [threading.main_thread().ident]


def test_collect_signal_before_native_ready_is_deferred():
    native = SimpleNamespace(_on_end=None, _end_pending=False)
    owner_thread_end(native)
    assert native._end_pending
