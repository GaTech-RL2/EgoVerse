import hashlib
import io
import json
import tarfile

import pytest

from reproductions.leact.cluster.efe_pear_asset_restore import restore_assets


def package(tmp_path):
    root=tmp_path/'package';root.mkdir()
    files={'release.json':b'{"version":1}', 'robot/arm.usda':b'#usda 1.0\n'}
    digests={name:hashlib.sha256(raw).hexdigest()for name,raw in files.items()}
    release={'release_manifest':'release.json','release_manifest_sha256':digests['release.json'],
        'patched_files':{'robot/arm.usda':{'sha256_after':digests['robot/arm.usda']}}}
    manifest={'release_manifest_sha256':digests['release.json'],
        'files':{name:{'bytes':len(raw),'sha256':digests[name]}for name,raw in files.items()}}
    (root/'asset-closure.json').write_text(json.dumps(manifest))
    with tarfile.open(root/'asset-closure.tar','w')as archive:
        for name,raw in files.items():
            member=tarfile.TarInfo(name);member.size=len(raw);archive.addfile(member,io.BytesIO(raw))
    return root,release


def test_restored_assets_match_the_release_and_reject_later_corruption(tmp_path):
    source,release=package(tmp_path);target=tmp_path/'assets'
    assert restore_assets(source,target,release)['passed']is True
    (target/'robot/arm.usda').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum mismatch'):restore_assets(source,target,release)


def test_asset_restore_rejects_unsafe_archive_members_before_publication(tmp_path):
    source,release=package(tmp_path);target=tmp_path/'assets'
    with tarfile.open(source/'asset-closure.tar','a')as archive:
        member=tarfile.TarInfo('../escape');member.size=3;archive.addfile(member,io.BytesIO(b'bad'))
    with pytest.raises(ValueError,match='Unexpected native asset archive member'):restore_assets(source,target,release)
    assert not target.exists()
    assert not(tmp_path/'escape').exists()


def test_asset_restore_requires_the_original_admitted_patch_digest(tmp_path):
    source,release=package(tmp_path);target=tmp_path/'assets'
    release['patched_files']['robot/arm.usda']['sha256_after']='0'*64
    with pytest.raises(ValueError,match='pinned native patch'):restore_assets(source,target,release)
    assert not target.exists()
