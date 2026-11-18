"""Resumable OCI publication with renewed, repository-scoped credentials.

Large native layers can outlive a registry token. Read the current short-lived
token per chunk and retain upload offsets across owned allocation retries.
Native config and layer bytes are never recompressed or changed.
"""
import base64
import hashlib
import http.client
import json
from pathlib import Path
import re
import tarfile
import time
from urllib.parse import urljoin,urlsplit,urlunsplit,parse_qsl,urlencode

from efe_pear_file_rpc import atomic_json

REPOSITORY='nvidian/lpr/pear-efe-rtx-rp-jhh'
ORIGIN='https://nvcr.io'
CHUNK_BYTES=512<<20


class RegistryError(RuntimeError):pass


class TokenFile:
    def __init__(self,path):self.path=Path(path)

    def __call__(self):
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            if self.path.exists():
                if self.path.stat().st_mode&0o077:raise ValueError('Registry token must be private')
                token=self.path.read_text().strip()
                try:
                    middle=token.split('.')[1]
                    claims=json.loads(base64.urlsafe_b64decode(middle+'='*(-len(middle)%4)))
                    if not any(v.get('name')==REPOSITORY and'push'in v.get('actions',[])for v in claims.get('access',[])):
                        raise ValueError('Token does not authorize the private experiment repository')
                    if claims['exp']>time.time()+90:return token
                except (IndexError,json.JSONDecodeError):pass
            time.sleep(1)
        raise TimeoutError('No fresh repository-scoped registry token arrived')


class Registry:
    def __init__(self,token):self.token=token;self.connection=None

    @staticmethod
    def location(value):
        result=urljoin(ORIGIN,value);parsed=urlsplit(result)
        if parsed.scheme!='https'or parsed.netloc!='nvcr.io'or not parsed.path.startswith('/v2/'+REPOSITORY+'/'):
            raise ValueError('Registry upload location left the authorized private repository')
        return result

    def request(self,method,location,body=b'',headers=None):
        parsed=urlsplit(self.location(location));target=parsed.path+('?'+parsed.query if parsed.query else'')
        if self.connection is None:self.connection=http.client.HTTPSConnection('nvcr.io',timeout=180)
        supplied={'Authorization':'Bearer '+self.token(),'Content-Length':str(len(body)),**(headers or{})}
        try:
            self.connection.request(method,target,body=body,headers=supplied)
            response=self.connection.getresponse();raw=response.read()
            return response.status,{k.lower():v for k,v in response.getheaders()},raw
        except (OSError,http.client.HTTPException):
            self.connection.close();self.connection=None
            raise RegistryError('Registry connection interrupted; resume uses the verified remote offset')from None


def digest_location(location,digest):
    parsed=urlsplit(location);query=parse_qsl(parsed.query,keep_blank_values=True)
    if any(k=='digest'for k,_ in query):raise ValueError('Unexpected digest in an open upload URL')
    return urlunsplit((parsed.scheme,parsed.netloc,parsed.path,urlencode([*query,('digest',digest)]),''))


def head_blob(registry,endpoint):
    code,headers,raw=registry.request('HEAD',endpoint)
    if code not in(302,307):return code,headers,raw
    parsed=urlsplit(headers.get('location',''))
    # NGC serves content through a signed layers URL. Never forward the
    # repository bearer token to that separate host.
    if (parsed.scheme!='https'or parsed.hostname!='layers.nvcr.io'or
        parsed.port not in(None,443)or parsed.username or parsed.password):
        raise ValueError('Unexpected native registry content redirect')
    connection=http.client.HTTPSConnection(parsed.hostname,timeout=60)
    target=parsed.path+('?'+parsed.query if parsed.query else'')
    try:
        connection.request('HEAD',target)
        response=connection.getresponse();metadata={k.lower():v for k,v in response.getheaders()}
        result=response.status;response.read()
    finally:connection.close()
    if'docker-content-digest'in headers:metadata['docker-content-digest']=headers['docker-content-digest']
    return result,metadata,b''


def push_blob(registry,stream,size,digest,state_file,*,chunk_bytes=CHUNK_BYTES):
    for attempt in range(6):
        try:
            return _push_blob_once(registry,stream,size,digest,state_file,chunk_bytes=chunk_bytes)
        except RegistryError as exc:
            if attempt==5:raise
            # A failed PATCH may already have reached the registry. Re-enter
            # through HEAD/GET and seek to its acknowledged offset; replaying
            # the same PATCH can duplicate or corrupt an otherwise valid layer.
            print('PEAR_NATIVE_REGISTRY_RETRY '+json.dumps({'digest':digest,'attempt':attempt+1,'reason':str(exc)}),flush=True)
            time.sleep(min(5*2**attempt,45))


def _push_blob_once(registry,stream,size,digest,state_file,*,chunk_bytes):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',digest):raise ValueError('Invalid native blob digest')
    endpoint='/v2/'+REPOSITORY+'/blobs/'+digest
    code,headers,_=head_blob(registry,endpoint)
    if code==200:
        if headers.get('docker-content-digest',digest)!=digest or int(headers['content-length'])!=size:
            raise ValueError('Existing registry blob has the wrong digest or size')
        Path(state_file).unlink(missing_ok=True)
        return
    if code!=404:raise RegistryError('Registry blob HEAD returned '+str(code))
    state_file=Path(state_file);state=json.loads(state_file.read_text())if state_file.exists()else None
    location=None;offset=0
    if state:
        if state['digest']!=digest or state['size']!=size:raise ValueError('Resumed upload belongs to different native bytes')
        code,headers,_=registry.request('GET',state['location'])
        if code in(202,204):
            location=registry.location(headers['location'])
            uploaded=headers.get('range','0-0').removeprefix('bytes=')
            if not re.fullmatch(r'0-[0-9]+',uploaded):raise ValueError('Registry upload status has an invalid range')
            offset=int(uploaded.split('-')[1])+1
            if uploaded=='0-0':
                # Registries use 0-0 for an empty upload as well as a single
                # accepted byte. Never invent offset 1 after a failed first
                # chunk. A new upload preserves all committed layers and
                # removes this ambiguous, uncommitted session from the path.
                location=None;offset=0
        elif code!=404:raise RegistryError('Registry upload status returned '+str(code))
    if location is None:
        code,headers,_=registry.request('POST','/v2/'+REPOSITORY+'/blobs/uploads/')
        if code!=202:raise RegistryError('Registry upload creation returned '+str(code))
        location=registry.location(headers['location'])
    if not 0<=offset<=size:raise ValueError('Registry upload offset exceeds the native blob')
    state={'digest':digest,'size':size,'location':location,'offset':offset}
    atomic_json(state_file,state);stream.seek(offset)
    if size<=chunk_bytes:
        # A short, fresh-token request also avoids the historical empty-range
        # ambiguity for tiny OCI configs and empty layers.
        if offset:raise ValueError('Unexpected partial upload of a monolithic small blob')
        code,headers,_=registry.request('PUT',digest_location(location,digest),stream.read(),
            {'Content-Type':'application/octet-stream'})
    else:
        while offset<size:
            block=stream.read(min(chunk_bytes,size-offset))
            if not block:raise ValueError('Native OCI blob ended before its declared size')
            end=offset+len(block)-1
            code,headers,_=registry.request('PATCH',location,block,{'Content-Type':'application/octet-stream',
                'Content-Range':f'{offset}-{end}'})
            if code!=202:raise RegistryError('Registry chunk upload returned '+str(code))
            if headers.get('range','').removeprefix('bytes=')!=f'0-{end}':
                raise ValueError('Registry acknowledged a different native byte range')
            location=registry.location(headers['location']);offset=end+1
            state.update(location=location,offset=offset);atomic_json(state_file,state)
        code,headers,_=registry.request('PUT',digest_location(location,digest))
    if code!=201:raise RegistryError('Registry native blob commit returned '+str(code))
    if headers.get('docker-content-digest',digest)!=digest:raise ValueError('Committed native blob digest differs')
    code,headers,_=head_blob(registry,endpoint)
    if code!=200 or int(headers['content-length'])!=size or headers.get('docker-content-digest',digest)!=digest:
        raise ValueError('Committed native blob failed digest/size readback: '+json.dumps({
            'status':code,'reported_bytes':headers.get('content-length'),'expected_bytes':size,
            'digest_matches':headers.get('docker-content-digest',digest)==digest}))
    state_file.unlink(missing_ok=True)


def publish(archive,expected_sha256,destination,token_file,receipt,*,archive_verified=False):
    archive=Path(archive);receipt=Path(receipt)
    prefix='docker://nvcr.io/'+REPOSITORY+':'
    if not destination.startswith(prefix):raise ValueError('Unapproved native image destination')
    tag=destination[len(prefix):]
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',tag):raise ValueError('Invalid native image tag')
    if not archive_verified:
        digest=hashlib.sha256()
        with archive.open('rb')as stream:
            while block:=stream.read(8<<20):digest.update(block)
        if digest.hexdigest()!=expected_sha256:raise ValueError('Pinned native OCI archive changed')
    registry=Registry(TokenFile(token_file));states=receipt.parent/'pending-uploads';states.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive)as container:
        raw=container.extractfile('index.json').read();manifest=json.loads(raw)
        while'manifests'in manifest:
            if len(manifest['manifests'])!=1:raise ValueError('Expected exactly one native image')
            descriptor=manifest['manifests'][0]
            raw=container.extractfile('blobs/'+descriptor['digest'].replace(':','/')).read()
            if 'sha256:'+hashlib.sha256(raw).hexdigest()!=descriptor['digest']:
                raise ValueError('Native manifest digest differs from OCI index')
            manifest=json.loads(raw)
        for descriptor in [manifest['config'],*manifest['layers']]:
            member=container.getmember('blobs/'+descriptor['digest'].replace(':','/'))
            if member.size!=descriptor['size']:raise ValueError('Native layer size differs from its manifest')
            with container.extractfile(member)as stream:
                push_blob(registry,stream,member.size,descriptor['digest'],states/(descriptor['digest'].replace(':','_')+'.json'))
            print('PEAR_NATIVE_BLOB_VERIFIED '+json.dumps({'digest':descriptor['digest'],'bytes':member.size}),flush=True)
    media=manifest.get('mediaType','application/vnd.oci.image.manifest.v1+json')
    endpoint='/v2/'+REPOSITORY+'/manifests/'+tag
    code,_,_=registry.request('PUT',endpoint,raw,{'Content-Type':media})
    if code!=201:raise RegistryError('Registry image manifest commit returned '+str(code))
    code,_,downloaded=registry.request('GET',endpoint,headers={'Accept':media})
    if code!=200 or downloaded!=raw:raise ValueError('Native image manifest failed exact readback')
    saved={'image':'nvcr.io/'+REPOSITORY+'@sha256:'+hashlib.sha256(downloaded).hexdigest(),
        'tag':destination.removeprefix('docker://'),'archive_sha256':expected_sha256,
        'archive_bytes':archive.stat().st_size,'registry_manifest_readback_verified':True,
        'config_digest':manifest['config']['digest'],'layer_digests':[v['digest']for v in manifest['layers']],
        'config_and_layers_unchanged':True,'private_registry':True,'primary_rl_updates':0,
        'credential_mode':'rotating_short_lived_repository_scoped_token_per_chunk'}
    atomic_json(receipt,saved);print('PEAR_NATIVE_IMAGE_PUBLISHED '+json.dumps(saved),flush=True)
    return saved
