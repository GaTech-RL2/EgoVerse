import io
import json
from pathlib import Path

import pytest

from reproductions.leact.cluster import efe_pear_object_archive as archive


class Store:
    def __init__(self):
        self.objects = {}; self.parts = {}; self.commits = []; self.aborts = []
        self.corrupt = False; self.fail_part = False

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        if kwargs.get('IfNoneMatch') == '*' and Key in self.objects:
            raise ValueError('Already committed')
        self.objects[Key] = Body
        if Key.endswith(archive.MANIFEST):
            self.commits.append(Key)

    def get_object(self, *, Bucket, Key):
        raw = self.objects[Key]
        if self.corrupt and not Key.endswith(archive.MANIFEST):
            raw += b'corrupted'
        return {'Body': io.BytesIO(raw)}

    def create_multipart_upload(self, **kwargs):
        self.parts[kwargs['Key']] = []
        return {'UploadId': 'owned-upload'}

    def upload_part(self, **kwargs):
        if self.fail_part:
            raise OSError('Interrupted transfer')
        self.parts[kwargs['Key']].append(kwargs['Body'])
        return {'ETag': str(kwargs['PartNumber'])}

    def complete_multipart_upload(self, **kwargs):
        self.objects[kwargs['Key']] = b''.join(self.parts.pop(kwargs['Key']))

    def abort_multipart_upload(self, **kwargs):
        self.aborts.append(kwargs['UploadId'])


def checkpoint(tmp_path):
    source = tmp_path / 'checkpoint'; (source / 'actor').mkdir(parents=True)
    (source / 'actor' / 'weights').write_bytes(bytes(range(255)) * 7)
    (source / 'curriculum.json').write_text('{"step":1}\n')
    (source / 'empty').touch()
    return source


def test_checkpoint_roundtrip_with_multipart_and_atomic_commit(tmp_path):
    store = Store(); source = checkpoint(tmp_path)
    saved = archive.archive(source, 's3://private/checkpoints/rp_jhh_step1', client=store,
                            workers=2, part_bytes=128)
    assert saved['readback_verified']
    assert len(store.commits) == 1 and not store.parts
    restored = tmp_path / 'restored'
    archive.restore('s3://private/checkpoints/rp_jhh_step1', restored, client=store)
    for path in source.rglob('*'):
        if path.is_file():
            assert (restored / path.relative_to(source)).read_bytes() == path.read_bytes()


def test_corrupt_readback_never_commits_checkpoint(tmp_path):
    store = Store(); store.corrupt = True
    with pytest.raises(ValueError, match='readback'):
        archive.archive(checkpoint(tmp_path), 's3://private/checkpoints/rp_jhh_step1',
                        client=store, part_bytes=128)
    assert not store.commits


def test_interrupted_multipart_upload_is_aborted(tmp_path):
    store = Store(); store.fail_part = True
    with pytest.raises(OSError, match='Interrupted'):
        archive.archive(checkpoint(tmp_path), 's3://private/checkpoints/rp_jhh_step1',
                        client=store, part_bytes=128)
    assert store.aborts == ['owned-upload'] and not store.commits


def test_unsafe_restore_member_leaves_no_partial_checkpoint(tmp_path):
    store = Store()
    store.objects['private/rp_jhh/' + archive.MANIFEST] = json.dumps({
        'readback_verified': True, 'object_prefix': 'private/rp_jhh/objects/attempt',
        'files': {'../outside': {'bytes': 1, 'sha256': 'unused', 'object_key': 'outside'}}}).encode()
    with pytest.raises(ValueError, match='Unsafe'):
        archive.restore('s3://bucket/private/rp_jhh', tmp_path / 'restore', client=store)
    assert not (tmp_path / 'restore').exists()
    assert not list(tmp_path.glob('*.restore-*'))
