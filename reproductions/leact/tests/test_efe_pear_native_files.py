import io
import tarfile

import pytest

from reproductions.leact.cluster.efe_pear_native_files import unpack_files


@pytest.mark.parametrize('name',['../escape.py','/outside.py'])
def test_native_source_layer_cannot_escape_its_output(tmp_path,name):
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w:gz')as archive:
        item=tarfile.TarInfo(name);item.size=1;archive.addfile(item,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='Unsafe'):unpack_files(raw.getvalue(),tmp_path)


def test_native_source_layer_keeps_exact_bytes_and_rejects_conflicting_replacement(tmp_path):
    def layer(data):
        raw=io.BytesIO()
        with tarfile.open(fileobj=raw,mode='w:gz')as archive:
            item=tarfile.TarInfo('sim/server.py');item.size=len(data);archive.addfile(item,io.BytesIO(data))
        return raw.getvalue()
    unpack_files(layer(b'original'),tmp_path)
    assert(tmp_path/'sim/server.py').read_bytes()==b'original'
    with pytest.raises(ValueError,match='Conflicting'):unpack_files(layer(b'changed'),tmp_path)
