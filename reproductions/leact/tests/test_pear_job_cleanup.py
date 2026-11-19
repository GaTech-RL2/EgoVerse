import os
import signal
import subprocess
import sys

import pytest

from reproductions.leact.cluster import efe_pear_job_cleanup as cleanup


def fake_process(root,pid,workflow='owned',control='/allocations/owned',attempt='old',started=100):
    p=root/str(pid);p.mkdir()
    fields=['S','1','1','1']+['0']*15+[str(started)]
    (p/'stat').write_text(str(pid)+' (process with spaces) '+' '.join(fields))
    env={'EFE_WORKFLOW_ID':workflow,'PEAR_ALLOCATION_CONTROL':control,'EFE_PEAR_ATTEMPT_ID':attempt}
    (p/'environ').write_bytes(b'\0'.join(k.encode()+b'='+v.encode()for k,v in env.items()))
    return p


def test_cleanup_is_scoped_to_the_allocation_and_excludes_the_current_job_and_keeper(tmp_path):
    fake_process(tmp_path,900001)
    fake_process(tmp_path,900002,attempt='new')
    fake_process(tmp_path,900003,workflow='someone-else')
    fake_process(tmp_path,900004,control='/different/owned')
    fake_process(tmp_path,900005,attempt='')
    assert set(cleanup.owned_processes('owned','/allocations/owned',keep_attempt='new',proc=tmp_path))=={900001}
    assert set(cleanup.owned_processes('owned','/allocations/owned',only_attempt='new',proc=tmp_path))=={900002}
    with pytest.raises(ValueError,match='exact allocation'):cleanup.owned_processes('owned','/allocations/other',proc=tmp_path)


def test_a_reused_pid_is_never_signaled(tmp_path,monkeypatch):
    fake_process(tmp_path,900001,started=200)
    monkeypatch.setattr(cleanup.os,'kill',lambda *a:pytest.fail('A reused PID was signaled'))
    assert not cleanup.signal_process(900001,{'started':100},signal.SIGKILL,proc=tmp_path)


@pytest.mark.skipif(not sys.platform.startswith('linux'),reason='Linux process ownership uses procfs')
def test_cleanup_reaps_a_detached_owned_child_and_preserves_an_unrelated_process():
    env={**os.environ,'EFE_WORKFLOW_ID':'cleanup-test','PEAR_ALLOCATION_CONTROL':'/allocations/cleanup-test',
        'EFE_PEAR_ATTEMPT_ID':'old'}
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],env=env,start_new_session=True)
    other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],env={**env,'EFE_WORKFLOW_ID':'unrelated'})
    try:
        value=cleanup.cleanup('cleanup-test','/allocations/cleanup-test',keep_attempt='new',grace_seconds=.5)
        child.wait(timeout=5)
        assert value['remaining']==0 and other.poll()is None
    finally:
        for p in [child,other]:
            if p.poll()is None:p.kill()
            p.wait()
