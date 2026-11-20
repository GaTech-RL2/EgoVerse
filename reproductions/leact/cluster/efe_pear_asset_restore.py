"""Restore the pinned native dependency closure into a private pool-local cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
from uuid import uuid4


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while block:=stream.read(8<<20):digest.update(block)
    return digest.hexdigest()


def verify_assets(root,manifest,release):
    root=Path(root)
    if manifest['release_manifest_sha256']!=release['release_manifest_sha256']:
        raise ValueError('Restored asset release differs from the admitted release')
    for name,row in manifest['files'].items():
        relative=PurePosixPath(name)
        if relative.is_absolute()or any(v in {'.','..'}for v in relative.parts):
            raise ValueError('Unsafe native asset member')
        path=root/name
        if path.is_symlink()or not path.is_file()or(path.stat().st_size,sha(path))!=(row['bytes'],row['sha256']):
            raise ValueError('Restored asset checksum mismatch: '+name)
    pins={release['release_manifest']:release['release_manifest_sha256'],
        **{name:row['sha256_after']for name,row in release['patched_files'].items()}}
    for name,digest in pins.items():
        if name not in manifest['files']or sha(root/name)!=digest:
            raise ValueError('Restored pinned native patch differs: '+name)


def restore_assets(package,destination,release):
    package=Path(package);destination=Path(destination)
    manifest=json.loads((package/'asset-closure.json').read_text())
    if not destination.exists():
        temporary=destination.with_name('.'+destination.name+'.'+uuid4().hex)
        temporary.mkdir(mode=0o700,parents=True)
        try:
            with tarfile.open(package/'asset-closure.tar')as archive:
                seen=set()
                for member in archive:
                    name=member.name;relative=PurePosixPath(name)
                    if(not member.isfile()or relative.is_absolute()or any(v in {'.','..'}for v in relative.parts)
                        or name not in manifest['files']or name in seen):
                        raise ValueError('Unexpected native asset archive member')
                    seen.add(name);target=temporary/name;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
                    with archive.extractfile(member)as source,target.open('xb')as output:
                        shutil.copyfileobj(source,output,8<<20)
                    target.chmod(0o600)
                if seen!=set(manifest['files']):raise ValueError('Native asset archive is incomplete')
            verify_assets(temporary,manifest,release);temporary.replace(destination)
        finally:shutil.rmtree(temporary,ignore_errors=True)
    verify_assets(destination,manifest,release)
    return {'assets_root':str(destination),'files':len(manifest['files']),
        'bytes':sum(row['bytes']for row in manifest['files'].values()),
        'release_manifest_sha256':release['release_manifest_sha256'],'passed':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,default=Path(__file__).with_name('spec.json'))
    spec=json.loads(parser.parse_args().spec.read_text())
    raw=os.environ.pop('RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION')
    root=Path(spec['root']);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    if root.stat().st_mode&0o077:raise ValueError('Pool-local staging root must be private')
    from efe_pear_file_rpc import atomic_json
    def status(phase,**fields):
        value={'phase':phase,'updated_unix':time.time(),**fields}
        atomic_json(root/'restore-status.json',value);print('PEAR_L40S_RESTORE '+json.dumps(value),flush=True)
    credential=Path('/dev/shm/rp_jhh_asset_restore_credentials.json')
    descriptor=os.open(credential,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w')as stream:stream.write(raw)
    os.environ['PEAR_OBJECTSTORE_CREDENTIAL_FILE']=str(credential)
    try:
        deps=Path('/tmp/rp_jhh_asset_restore_packages')
        subprocess.run([sys.executable,'-m','pip','install','--quiet','--target',str(deps),'boto3==1.44.5'],check=True,timeout=300)
        sys.path.insert(0,str(deps));from efe_pear_object_archive import restore,configured_client,split_uri,MANIFEST
        package=root/'asset-package'
        status('restoring_asset_archive')
        client=configured_client();bucket,prefix=split_uri(spec['source'])
        deadline=time.monotonic()+spec.get('archive_wait_seconds',0)
        while True:
            try:client.head_object(Bucket=bucket,Key=prefix+'/'+MANIFEST);break
            except Exception as exc:
                code=getattr(exc,'response',{}).get('Error',{}).get('Code')
                if code not in {'NoSuchKey','404','NotFound'}or time.monotonic()>=deadline:raise
                time.sleep(5)
        restore(spec['source'],package,client=client)
        receipt=restore_assets(package,spec['assets_root'],spec['release'])
        atomic_json(root/'asset-restore-receipt.json',receipt);status('assets_restored',**receipt)
        sources=Path(spec['sources_root']);sources.mkdir(mode=0o700,parents=True,exist_ok=True)
        deadline=time.monotonic()+spec.get('source_wait_seconds',1800)
        while time.monotonic()<deadline:
            missing=[name for name,row in spec['sources'].items()if not(sources/name).is_file()]
            if not missing:break
            status('waiting_for_source_payloads',missing=len(missing));time.sleep(15)
        else:raise TimeoutError('Pool-local source payloads have not arrived')
        for name,row in spec['sources'].items():
            path=sources/name
            if(path.stat().st_size,sha(path))!=(row['bytes'],row['sha256']):raise ValueError('Pool-local source payload changed')
        atomic_json(root/'source-restore-receipt.json',{'passed':True,'files':spec['sources']})
        status('pool_local_assets_and_sources_verified',**receipt)
        time.sleep(spec.get('inspection_seconds',300))
    finally:credential.unlink(missing_ok=True)


if __name__=='__main__':main()
