import hashlib
import json
from pathlib import Path

import pytest

from reproductions.leact.cluster import efe_approved_optimizer_cleanup as cleanup


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(cleanup,'ROOT',tmp_path)
    monkeypatch.setattr(cleanup,'AUDIT',tmp_path/'receipt/cleanup.json')
    files=[]
    for name in sorted(cleanup.APPROVED):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'old optimizer state')
        (path.parent/'model.safetensors').write_bytes(b'preserved model weights')
        (path.parent/'checkpoint-manifest.json').write_text(json.dumps({'files':{
            'model.safetensors':{'bytes':23,'sha256':hashlib.sha256(b'preserved model weights').hexdigest()}}}))
        s=path.stat();files.append({'path':str(path),'inode':s.st_ino,'bytes':s.st_size,'links':s.st_nlink})
    protected=tmp_path/'source/train/checkpoint-32/optimizer.pt';protected.parent.mkdir(parents=True)
    protected.write_bytes(b'required Adam state')
    (protected.parent/'model.safetensors').write_bytes(b'step32 weights')
    (protected.parent/'checkpoint-manifest.json').write_text(json.dumps({'files':{'model.safetensors':{'bytes':14}}}))
    proposal={'status':'approved_by_user','approved_utc':'fixture','files':files,
              'preserved_optimizer_files':[str(protected)]}
    return proposal,protected


def test_delete_only_approved_files_preserve_models_and_resume_idempotently(tmp_path,monkeypatch):
    proposal,protected=fixture(tmp_path,monkeypatch)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*.safetensors')}
    receipt=cleanup.execute(proposal)
    assert receipt['status']=='completed' and len(receipt['removed'])==5
    assert all(not Path(row['path']).exists() for row in proposal['files'])
    assert protected.read_bytes()==b'required Adam state'
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*.safetensors')}
    assert cleanup.execute(proposal)['removed']==receipt['removed']


def test_changed_candidate_aborts_before_any_deletion(tmp_path,monkeypatch):
    proposal,_=fixture(tmp_path,monkeypatch)
    Path(proposal['files'][-1]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity changed'):cleanup.execute(proposal)
    assert all(Path(row['path']).exists() for row in proposal['files'])


def test_wrong_scope_or_missing_approval_never_deletes(tmp_path,monkeypatch):
    proposal,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='approval'):cleanup.execute({**proposal,'status':'awaiting_user_approval'})
    with pytest.raises(ValueError,match='five-file scope'):cleanup.execute({**proposal,'files':proposal['files'][:4]})
    assert all(Path(row['path']).exists() for row in proposal['files'])
