import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cluster'))
from efe_pear_queue_checkpoint import shared_queue_checkpoint


def test_queue_roundtrip_uses_shared_paths_and_preserves_native_file_layout(tmp_path,monkeypatch):
    monkeypatch.setattr('efe_pear_checkpoint_archive.shutil.disk_usage',lambda p:SimpleNamespace(free=1<<40))
    shared=tmp_path/'shared';local=tmp_path/'head-only'/'transfer_queue';local.parent.mkdir()
    seen=[]
    def save(path,**kwargs):
        path=Path(path);assert path.is_relative_to(shared)
        (path/'storage_units').mkdir(parents=True)
        (path/'controller.pkl').write_bytes(b'controller state')
        (path/'storage_units/su_0.pkl').write_bytes(b'worker zero payload')
        (path/'storage_units/su_1.pkl').write_bytes(b'worker one payload')
        seen.append(('save',kwargs['metadata']))
    def load(path,**kwargs):
        path=Path(path);assert path.is_relative_to(shared)
        assert(path/'storage_units/su_0.pkl').read_bytes()==b'worker zero payload'
        assert(path/'storage_units/su_1.pkl').read_bytes()==b'worker one payload'
        assert not(path/'rp_jhh_archive_manifest.json').exists()
        seen.append(('load',None))
    tq=SimpleNamespace(save_checkpoint=save,load_checkpoint=load)
    with shared_queue_checkpoint(tq,shared):
        tq.save_checkpoint(str(local),metadata={'step':4})
        assert(local/'controller.pkl').read_bytes()==b'controller state'
        tq.load_checkpoint(str(local))
    assert tq.save_checkpoint is save and tq.load_checkpoint is load
    assert seen==[('save',{'step':4}),('load',None)]and not list(shared.iterdir())


def test_failed_queue_write_restores_api_and_does_not_publish_a_partial_copy(tmp_path):
    shared=tmp_path/'shared';local=tmp_path/'head-only'/'transfer_queue'
    def failed(path,**kwargs):
        Path(path).mkdir(parents=True);(Path(path)/'partial').write_bytes(b'incomplete')
        raise OSError('interrupted distributed save')
    tq=SimpleNamespace(save_checkpoint=failed,load_checkpoint=lambda path:None)
    with pytest.raises(OSError,match='interrupted'):
        with shared_queue_checkpoint(tq,shared):tq.save_checkpoint(str(local))
    assert tq.save_checkpoint is failed and not local.exists()and not list(shared.iterdir())
