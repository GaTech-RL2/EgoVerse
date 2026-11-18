import hashlib
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cluster'))
from efe_pear_registry_http import push_blob,Registry,RegistryError,REPOSITORY


class RegistryDouble:
    location=staticmethod(Registry.location)

    def __init__(self):
        self.uploaded=b'';self.committed=False;self.bad_range=False;self.methods=[]

    def request(self,method,location,body=b'',headers=None):
        self.methods.append(method)
        url='https://nvcr.io/v2/'+REPOSITORY+'/blobs/uploads/private-id?_state=position'
        digest='sha256:'+hashlib.sha256(self.uploaded).hexdigest()
        if method=='HEAD':
            return (200,{'content-length':str(len(self.uploaded)),'docker-content-digest':digest},b'')if self.committed else(404,{},b'')
        if method=='POST':return 202,{'location':url},b''
        if method=='GET':return 204,{'location':url,'range':f'0-{len(self.uploaded)-1}'},b''
        if method=='PATCH':
            assert headers['Content-Range']==f'{len(self.uploaded)}-{len(self.uploaded)+len(body)-1}'
            self.uploaded+=body
            end=len(self.uploaded)-1+(1 if self.bad_range else 0)
            return 202,{'location':url,'range':f'0-{end}'},b''
        if method=='PUT':
            self.uploaded+=body;self.committed=True
            return 201,{'location':url,'docker-content-digest':'sha256:'+hashlib.sha256(self.uploaded).hexdigest()},b''
        raise AssertionError(method)


def test_native_blob_chunk_roundtrip_preserves_exact_bytes(tmp_path):
    payload=bytes(range(255))*37;digest='sha256:'+hashlib.sha256(payload).hexdigest();registry=RegistryDouble()
    state=tmp_path/'pending.json'
    push_blob(registry,io.BytesIO(payload),len(payload),digest,state,chunk_bytes=1024)
    assert registry.committed and registry.uploaded==payload and not state.exists()
    assert registry.methods.count('PATCH')>1


def test_upload_resume_uses_remote_offset_when_local_receipt_is_older(tmp_path):
    payload=bytes(range(255))*37;digest='sha256:'+hashlib.sha256(payload).hexdigest();registry=RegistryDouble()
    registry.uploaded=payload[:2048]
    state=tmp_path/'pending.json'
    state.write_text(json.dumps({'digest':digest,'size':len(payload),'offset':1024,
        'location':'https://nvcr.io/v2/'+REPOSITORY+'/blobs/uploads/private-id?_state=old'}))
    push_blob(registry,io.BytesIO(payload),len(payload),digest,state,chunk_bytes=1024)
    assert registry.uploaded==payload and registry.committed
    assert 'POST'not in registry.methods


def test_wrong_native_byte_range_never_commits_manifest(tmp_path):
    payload=b'exact native bytes'*100;registry=RegistryDouble();registry.bad_range=True
    with pytest.raises(ValueError,match='different native byte range'):
        push_blob(registry,io.BytesIO(payload),len(payload),'sha256:'+hashlib.sha256(payload).hexdigest(),
            tmp_path/'pending.json',chunk_bytes=128)
    assert not registry.committed


def test_registry_upload_does_not_forward_credentials_to_another_origin():
    with pytest.raises(ValueError,match='authorized private repository'):
        Registry.location('https://example.com/v2/'+REPOSITORY+'/blobs/uploads/id')


def test_server_error_after_accepting_chunk_resumes_without_repeating_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr('efe_pear_registry_http.time.sleep',lambda _:None)
    class AcceptedThenFailed(RegistryDouble):
        failed=False
        def request(self,method,*args,**kwargs):
            result=super().request(method,*args,**kwargs)
            if method=='PATCH'and not self.failed:
                self.failed=True
                return 500,{},b'temporary registry error'
            return result
    payload=bytes(range(255))*37;registry=AcceptedThenFailed()
    push_blob(registry,io.BytesIO(payload),len(payload),'sha256:'+hashlib.sha256(payload).hexdigest(),
        tmp_path/'pending.json',chunk_bytes=1024)
    assert registry.committed and registry.uploaded==payload
    assert registry.methods.count('POST')==1 and'GET'in registry.methods


def test_ambiguous_empty_upload_status_starts_a_fresh_session(tmp_path):
    class EmptyStatus(RegistryDouble):
        def request(self,method,*args,**kwargs):
            if method=='GET':
                self.methods.append(method)
                return 204,{'location':'https://nvcr.io/v2/'+REPOSITORY+'/blobs/uploads/empty','range':'0-0'},b''
            return super().request(method,*args,**kwargs)
    payload=b'pinned simulator bytes'*100;registry=EmptyStatus()
    state=tmp_path/'pending.json'
    state.write_text(json.dumps({'digest':'sha256:'+hashlib.sha256(payload).hexdigest(),
        'size':len(payload),'offset':1,'location':'https://nvcr.io/v2/'+REPOSITORY+'/blobs/uploads/empty'}))
    push_blob(registry,io.BytesIO(payload),len(payload),'sha256:'+hashlib.sha256(payload).hexdigest(),state,chunk_bytes=128)
    assert registry.uploaded==payload and registry.committed and'POST'in registry.methods
