from copy import deepcopy
import hashlib

import pytest

from reproductions.leact.cluster.efe_qwen36_leact import reuse_encoded_row


@pytest.fixture
def encoded(tmp_path):
    source=tmp_path/'original';destination=tmp_path/'prepared';path=source/'tensors/pair-forward.pt'
    path.parent.mkdir(parents=True);path.write_bytes(b'unchanged frozen serialized tensor')
    row={'id':'pair-forward','state_id':'pair','decision_id':'decision','direction':'forward',
         'tensor_file':'tensors/pair-forward.pt','tensor_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'tokens':100,'target_tokens':20}
    return source,destination,row


def test_identical_candidate_reuses_exact_verified_bytes_and_metadata(encoded):
    source,destination,row=encoded;before=deepcopy(row)
    assert reuse_encoded_row(source,destination,row,'decision','pair','forward')==row
    assert(destination/row['tensor_file']).samefile(source/row['tensor_file'])
    assert row==before
    assert reuse_encoded_row(source,destination,row,'decision','pair','forward')==row


@pytest.mark.parametrize('key,value',[('decision_id','other'),('direction','backward'),('state_id','other'),('id','other')])
def test_changed_candidate_identity_cannot_reuse_a_tensor(encoded,key,value):
    source,destination,row=encoded;row[key]=value
    with pytest.raises(ValueError,match='identity differs'):
        reuse_encoded_row(source,destination,row,'decision','pair','forward')
    assert not destination.exists()


def test_source_integrity_is_checked_even_for_an_existing_hardlink(encoded):
    source,destination,row=encoded
    reuse_encoded_row(source,destination,row,'decision','pair','forward')
    (source/row['tensor_file']).write_bytes(b'changed source bytes')
    with pytest.raises(ValueError,match='source checksum changed'):
        reuse_encoded_row(source,destination,row,'decision','pair','forward')
