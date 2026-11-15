from datetime import datetime,timezone
import fcntl
import hashlib
import json

import pytest

from reproductions.leact.cluster.efe_relocate_inactive_warmstart import relocate


def fixture(tmp_path):
    source=tmp_path/'old/warmstart'
    checkpoint=source/'restart-states/checkpoint-176';checkpoint.mkdir(parents=True)
    data=b'exact checkpoint bytes'
    (checkpoint/'optimizer.pt').write_bytes(data)
    receipt={'step':176,'target_step':737,'files':{'optimizer.pt':{
        'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}}
    (checkpoint/'training-state.json').write_text(json.dumps(receipt))
    (source/'dataset-manifest.json').write_text('{"unchanged":true}')
    proof={'utc':datetime.now(timezone.utc).isoformat(),'controller_stopped':True,'active_warmstart_workflows':[]}
    return source,tmp_path/'new/warmstart',tmp_path/'state',proof


def test_relocation_preserves_checkpoint_source_and_canonical_path(tmp_path):
    source,destination,state,proof=fixture(tmp_path)
    result=relocate(source,destination,state,proof)
    assert source.is_symlink() and source.resolve()==destination
    assert result['original_data_deleted'] is False
    backup=source.with_name('warmstart.before-storage-relocation-20261113')
    assert (backup/'restart-states/checkpoint-176/optimizer.pt').read_bytes()==(source/'restart-states/checkpoint-176/optimizer.pt').read_bytes()
    assert (backup/'dataset-manifest.json').read_bytes()==(source/'dataset-manifest.json').read_bytes()


def test_corrupted_checkpoint_cannot_replace_source(tmp_path):
    source,destination,state,proof=fixture(tmp_path)
    (source/'restart-states/checkpoint-176/optimizer.pt').write_bytes(b'bad')
    with pytest.raises(ValueError,match='Checkpoint receipt differs'):
        relocate(source,destination,state,proof)
    assert source.is_dir() and not source.is_symlink() and not destination.exists()


def test_live_writer_prevents_relocation(tmp_path):
    source,destination,state,proof=fixture(tmp_path)
    with (source/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):relocate(source,destination,state,proof)
    assert not source.is_symlink() and not destination.exists()
