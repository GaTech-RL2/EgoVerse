import errno
import hashlib
import json

import pytest

from reproductions.leact.cluster.efe_leact_vl import link_replay
from reproductions.leact.cluster.efe_leact_pipeline import export_portable
from reproductions.leact.cluster.collect_efe_model_chunks import assemble


def test_replay_cross_device_copy_and_unrelated_errors(tmp_path, monkeypatch):
    source=tmp_path/'source.pt';source.write_bytes(b'frozen tensor bytes')
    def cross_device(*args):raise OSError(errno.EXDEV,'cross-device link')
    monkeypatch.setattr('os.link',cross_device)
    target=tmp_path/'copy.pt';link_replay(source,target)
    assert target.read_bytes()==source.read_bytes()
    def disk_full(*args):raise OSError(errno.ENOSPC,'disk full')
    monkeypatch.setattr('os.link',disk_full)
    with pytest.raises(OSError) as e:link_replay(source,tmp_path/'failed.pt')
    assert e.value.errno==errno.ENOSPC


def test_portable_export_preserves_model_and_excludes_optimizer(tmp_path):
    source=tmp_path/'checkpoint-12';source.mkdir()
    weights=source/'model.safetensors';weights.write_bytes(b'verified weights')
    (source/'optimizer.pt').write_bytes(b'node local Adam')
    manifest=source/'checkpoint-manifest.json'
    manifest.write_text(json.dumps({'revision':'test-final','files':{'model.safetensors':{
        'bytes':weights.stat().st_size,'sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}}}))
    ready={'checkpoint':str(source),'manifest':str(manifest),'revision':'test-final','final':True}
    root=tmp_path/'run';root.mkdir()
    receipt=export_portable(root,ready)
    assert (root/'portable-model/model.safetensors').read_bytes()==weights.read_bytes()
    assert not (root/'portable-model/optimizer.pt').exists()
    assert (source/'optimizer.pt').exists()
    assert receipt['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError):export_portable(root,dict(ready,final=False))


def test_checkpoint_chunk_assembly_verifies_bytes_before_replacing(tmp_path):
    chunks=tmp_path/'chunks';chunks.mkdir()
    (chunks/'part-00001.bin').write_bytes(b' checkpoint')
    (chunks/'part-00000.bin').write_bytes(b'complete')
    expected=b'complete checkpoint'
    record={'bytes':len(expected),'sha256':hashlib.sha256(expected).hexdigest()}
    output=tmp_path/'model.safetensors'
    assemble(chunks,output,record)
    assert output.read_bytes()==expected
    (chunks/'part-00001.bin').write_bytes(b' corrupt!!!')
    with pytest.raises(ValueError,match='failed verification'):
        assemble(chunks,output,record)
    assert output.read_bytes()==expected
    assert not output.with_suffix('.safetensors.assembling').exists()
