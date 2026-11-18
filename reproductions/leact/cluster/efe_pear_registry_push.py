"""Publish an exact cached simulator OCI to an authorized private repository.

Accept only a short-lived, repository-scoped bearer token from a mode-0600
file. No long-lived registry credentials are copied to the compute worker.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time


def publish(archive,expected_sha256,destination,token_file,receipt,*,archive_verified=False):
    archive,token_file,receipt=map(Path,[archive,token_file,receipt])
    if not destination.startswith('docker://nvcr.io/nvidian/lpr/pear-efe-rtx-rp-jhh:'):
        raise ValueError('Unexpected private experiment repository')
    if token_file.stat().st_mode&0o077:raise ValueError('Registry token file must be private')
    token=token_file.read_text().strip();token_file.unlink()
    import base64
    middle=token.split('.')[1]
    claims=json.loads(base64.urlsafe_b64decode(middle+'='*(-len(middle)%4)))
    if claims['exp']-time.time()<120:raise ValueError('The scoped registry token expires too soon')
    repository=destination.removeprefix('docker://nvcr.io/').rsplit(':',1)[0]
    if not any(v.get('name')==repository and'push'in v.get('actions',[])for v in claims.get('access',[])):
        raise ValueError('The scoped token does not authorize this private image')
    if not archive_verified:
        digest=hashlib.sha256()
        with archive.open('rb')as stream:
            while chunk:=stream.read(16<<20):digest.update(chunk)
        if digest.hexdigest()!=expected_sha256:raise ValueError('Cached simulator OCI checksum changed')
    help_text=subprocess.check_output(['skopeo','copy','--help'],text=True)
    if '--dest-registry-token'not in help_text:raise ValueError('Installed skopeo lacks scoped bearer-token support')
    with tarfile.open(archive)as stream:
        index=json.load(stream.extractfile('index.json'))
        while 'manifests'in index:
            if len(index['manifests'])!=1:raise ValueError('Expected one cached native image')
            index=json.load(stream.extractfile('blobs/'+index['manifests'][0]['digest'].replace(':','/')))
        expected_config=index['config']['digest'];expected_layers=[v['digest']for v in index['layers']]
    preserve=['--preserve-digests']if '--preserve-digests'in help_text else[]
    process=subprocess.run(['skopeo','copy',*preserve,'--dest-registry-token',token,
        'oci-archive:'+str(archive),destination],capture_output=True,text=True)
    print((process.stdout+process.stderr).replace(token,'[REDACTED]')[-8000:],flush=True)
    if process.returncode:raise RuntimeError('Private simulator image publication failed; exit='+str(process.returncode))
    result=subprocess.run(['skopeo','inspect','--raw','--registry-token',token,destination],capture_output=True)
    if result.returncode:raise RuntimeError('Published simulator manifest readback failed')
    remote_manifest=json.loads(result.stdout)
    if remote_manifest['config']['digest']!=expected_config or [v['digest']for v in remote_manifest['layers']]!=expected_layers:
        raise ValueError('Registry image config/layers differ from the exact cached simulator')
    manifest_sha=hashlib.sha256(result.stdout).hexdigest()
    saved={'image':destination.removeprefix('docker://').rsplit(':',1)[0]+'@sha256:'+manifest_sha,
        'tag':destination.removeprefix('docker://'),'archive_sha256':expected_sha256,
        'archive_bytes':archive.stat().st_size,'registry_manifest_readback_verified':True,
        'config_digest':expected_config,'layer_digests':expected_layers,'config_and_layers_unchanged':True,
        'private_registry':True,'primary_rl_updates':0}
    receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(saved,indent=2)+'\n')
    print('PEAR_NATIVE_IMAGE_PUBLISHED '+json.dumps(saved),flush=True)
    return saved


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['archive','expected-sha256','destination','token-file','receipt']:parser.add_argument('--'+key,required=True)
    args=parser.parse_args();publish(args.archive,args.expected_sha256,args.destination,args.token_file,args.receipt)
