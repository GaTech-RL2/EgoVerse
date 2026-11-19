from pathlib import Path

import pytest


def test_native_infrastructure_retries_preserve_the_lane_but_source_errors_fail(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'cluster'))
    import efe_pear_remote_worker as worker
    class InfrastructureError(RuntimeError):pass
    root=tmp_path/'transport';root.mkdir(mode=0o700)
    finished=tmp_path/'finished.json';calls=[]
    monkeypatch.setattr(worker,'claim_episode',lambda *a,**k:(root,{'env_args':{}}))
    def episode(*args):
        calls.append(1)
        if len(calls)<=6:raise InfrastructureError('Native instance startup failed')
        finished.touch()
    monkeypatch.setattr(worker,'serve_episode',episode)
    original_wait=worker.threading.Event.wait
    monkeypatch.setattr(worker.threading.Event,'wait',lambda self,timeout=None:
        original_wait(self,.001 if timeout in {2,4,8,16,32,60}else timeout))
    worker.serve(root,'fixture-worker',{'test'},object,finished=finished,recoverable_errors=(InfrastructureError,))
    assert len(calls)==7
    finished.unlink();calls.clear()
    def broken(*args):calls.append(1);raise ValueError('Programming error')
    monkeypatch.setattr(worker,'serve_episode',broken)
    with pytest.raises(RuntimeError,match='repeatedly failed'):
        worker.serve(root,'fixture-worker',{'test'},object,finished=finished,recoverable_errors=(InfrastructureError,))
    assert len(calls)==5
