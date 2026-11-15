import base64
import hashlib
import json
import zlib

import pytest

from reproductions.leact.cluster.monitor_efe_astra import collection_worker,restore_local_export


def test_running_job_collects_from_its_own_allocation():
    assert collection_worker({'name':'current','status':'RUNNING'},[])=='current'


def test_completed_job_uses_only_a_running_allocation_owned_by_same_user():
    state={'name':'finished','status':'COMPLETED','submitted_by':'owner'}
    active=[{'name':'foreign','user':'other','status':'RUNNING'},
            {'name':'waiting','user':'owner','status':'PENDING'},
            {'name':'reader','user':'owner','status':'RUNNING'}]
    assert collection_worker(state,active)=='reader'
    assert collection_worker(state,active[:2]) is None


def test_missing_owner_does_not_authorize_using_another_allocation():
    assert collection_worker({'name':'finished','status':'COMPLETED'},
                             [{'name':'other','status':'RUNNING'}]) is None


def export_line(files):
    return 'OSMO log prefix EFE_LOCAL_RESULTS '+base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()


def test_node_local_results_survive_worker_exit_in_checksummed_logs(tmp_path):
    data=b'{"diagnostic_success":true}'
    item={'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()}
    result=restore_local_export(export_line({'complete.json':item}),tmp_path,['complete.json'])
    assert result=={'complete.json':item['sha256']} and (tmp_path/'complete.json').read_bytes()==data


@pytest.mark.parametrize('bad_name,bad_hash',[('../outside.json',False),('complete.json',True)])
def test_invalid_log_export_cannot_write_files(tmp_path,bad_name,bad_hash):
    data=b'{}';item={'sha256':'bad' if bad_hash else hashlib.sha256(data).hexdigest(),
                     'data':base64.b64encode(data).decode()}
    with pytest.raises(ValueError):restore_local_export(export_line({bad_name:item}),tmp_path,['complete.json'])
    assert not list(tmp_path.iterdir())
