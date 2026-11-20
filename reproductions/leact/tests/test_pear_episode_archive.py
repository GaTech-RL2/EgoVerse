import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cluster'))
from efe_pear_episode_archive import completed_episode


def test_unverified_native_archive_keeps_original_rollout(tmp_path,monkeypatch):
    monkeypatch.setenv('PEAR_EPISODE_ARCHIVE','s3://private/native')
    monkeypatch.setattr('efe_pear_object_archive.archive',lambda *a,**k:{'bytes':123,'readback_verified':False})
    (tmp_path/'rollout.mp4').write_bytes(b'original native frames')
    assert completed_episode(tmp_path,'episode')is None
    assert(tmp_path/'rollout.mp4').read_bytes()==b'original native frames'
    assert not(tmp_path/'rp_jhh_native_archive.json').exists()


def test_verified_native_archive_retains_local_verdict_and_archive_pointer(tmp_path,monkeypatch):
    monkeypatch.setenv('PEAR_EPISODE_ARCHIVE','s3://private/native')
    monkeypatch.setattr('efe_pear_object_archive.archive',lambda *a,**k:{'bytes':123,'readback_verified':True})
    (tmp_path/'rollout.mp4').write_bytes(b'original native frames')
    (tmp_path/'verdict.json').write_text(json.dumps({'success':True}))
    result=completed_episode(tmp_path,'episode')
    assert result['archive']=='s3://private/native/episode'and result['readback_verified']
    assert not(tmp_path/'rollout.mp4').exists()
    assert json.loads((tmp_path/'verdict.json').read_text())=={'success':True}
    assert json.loads((tmp_path/'rp_jhh_native_archive.json').read_text())==result
