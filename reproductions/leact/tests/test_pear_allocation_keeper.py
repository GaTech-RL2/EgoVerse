import hashlib
import json
import time

from reproductions.leact.cluster.efe_pear_allocation_keeper import Keeper,write


def request(root,bundle,ident):
    value={'id':ident,'bundle':str(bundle),'entry':'job.py','roles':['trainer'],
        'files':{name:hashlib.sha256((bundle/name).read_bytes()).hexdigest()for name in ['job.py','spec.json']}}
    write(root/'desired.json',value)


def wait_idle(keeper):
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        assert keeper.tick()
        if keeper.phase.startswith('idle_'):return
        time.sleep(.02)
    raise AssertionError('Child did not finish')


def test_failed_and_successful_jobs_leave_the_same_allocation_available(tmp_path):
    root=tmp_path/'allocation';bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'spec.json').write_text('{}')
    (bundle/'job.py').write_text('raise RuntimeError("training failed")\n')
    keeper=Keeper(root,'trainer',['trainer'],environment={})
    request(root,bundle,'failed-job');wait_idle(keeper)
    assert keeper.phase=='idle_failed'
    keeper_pid=json.loads((root/'nodes/trainer.json').read_text())['keeper_pid']
    (bundle/'job.py').write_text('from pathlib import Path\nPath(__file__).with_suffix(".done").touch()\n')
    request(root,bundle,'replacement-job');wait_idle(keeper)
    assert keeper.phase=='idle_completed'and(bundle/'job.done').exists()
    assert keeper.status()['keeper_pid']==keeper_pid
    assert keeper.tick()  # Success also leaves the allocation alive.
    write(root/'release.json',{'release':True,'authorized_by':'user'})
    assert keeper.tick()is False


def test_rejected_job_cannot_replace_a_running_process(tmp_path):
    root=tmp_path/'allocation';bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'spec.json').write_text('{}');(bundle/'job.py').write_text('pass\n')
    keeper=Keeper(root,'trainer',['trainer'],environment={})
    request(root,bundle,'bad');(bundle/'job.py').write_text('raise RuntimeError()\n')
    assert keeper.tick()and keeper.child is None
    assert json.loads((root/'rejected/bad.json').read_text())['error'].startswith('Allocation job source changed')


def test_one_failed_job_peer_stops_work_but_preserves_other_nodes(tmp_path):
    keeper=Keeper(tmp_path/'allocation','trainer',['trainer','rollout-0'],environment={})
    class Child:
        pid=123
        def poll(self):return None
    keeper.child=Child();keeper.job='job-1';keeper.phase='running'
    write(keeper.root/'nodes/rollout-0.json',{'job_id':'job-1','phase':'idle_failed'})
    # No real process is signaled by this hermetic process-double check.
    from unittest.mock import patch
    with patch('reproductions.leact.cluster.efe_pear_allocation_keeper.stop_child')as stop:
        assert keeper.tick()
    stop.assert_called_once_with(keeper.child)
    assert keeper.phase=='idle_peer_failed'
