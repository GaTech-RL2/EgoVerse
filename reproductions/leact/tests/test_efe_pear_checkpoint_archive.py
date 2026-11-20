from pathlib import Path

import pytest

from reproductions.leact.cluster.efe_pear_checkpoint_archive import archive


def test_archive_keeps_source_and_verifies_idempotent_publication(tmp_path):
    source=tmp_path/'local';(source/'actor').mkdir(parents=True)
    (source/'actor/model.pt').write_bytes(b'model'*200)
    (source/'curriculum.json').write_text('{"step":1}')
    destination=tmp_path/'shared/rp_jhh_step1'
    first=archive(source,destination,reserve_bytes=0)
    assert first['readback_verified']is True
    assert set(first['files'])=={'actor/model.pt','curriculum.json'}
    assert (source/'actor/model.pt').exists()
    assert archive(source,destination,reserve_bytes=0)==first
    (destination/'actor/model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='differs'):archive(source,destination,reserve_bytes=0)


def test_archive_refuses_unbounded_paths_and_missing_content(tmp_path):
    source=tmp_path/'local';source.mkdir()
    with pytest.raises(ValueError,match='empty'):archive(source,tmp_path/'dest',reserve_bytes=0)
    (source/'link').symlink_to(Path(__file__))
    with pytest.raises(ValueError,match='symbolic'):archive(source,tmp_path/'dest',reserve_bytes=0)


def test_retried_publication_rejects_a_different_optimizer_checkpoint(tmp_path):
    source=tmp_path/'local';source.mkdir();(source/'model').write_bytes(b'first')
    target=tmp_path/'shared';archive(source,target,reserve_bytes=0)
    (source/'model').write_bytes(b'other')
    with pytest.raises(ValueError,match='differs'):archive(source,target,reserve_bytes=0)
    (source/'model').write_bytes(b'first');(source/'optimizer').write_bytes(b'new state')
    with pytest.raises(ValueError,match='differs'):archive(source,target,reserve_bytes=0)


def test_failed_readback_removes_only_its_partial_copy(tmp_path,monkeypatch):
    from reproductions.leact.cluster import efe_pear_checkpoint_archive as module
    source=tmp_path/'local';source.mkdir();(source/'model').write_bytes(b'keep source')
    destination=tmp_path/'shared/checkpoint';unrelated=tmp_path/'shared/.other.partial'
    unrelated.mkdir(parents=True);(unrelated/'file').write_text('preserve')
    monkeypatch.setattr(module,'digest',lambda path:'wrong checksum')
    with pytest.raises(ValueError,match='readback'):archive(source,destination,reserve_bytes=0)
    assert (source/'model').read_bytes()==b'keep source'
    assert (unrelated/'file').read_text()=='preserve'
    assert list(destination.parent.iterdir())==[unrelated]
