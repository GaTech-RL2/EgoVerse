"""Copy the exact USD dependency closure needed by the admitted native tasks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while block:=stream.read(8<<20):value.update(block)
    return value.hexdigest()


def task_owned_files(root,roots):
    """Include task manifests and runtime sidecars absent from USD references."""
    root=Path(root).resolve();prefix=('scenes','YAM','eval_flywheel','T_gen')
    directories={Path(*Path(name).parts[:5])for name in roots
        if Path(name).parts[:4]==prefix and len(Path(name).parts)>=5}
    files=set()
    for relative in sorted(directories):
        if '..'in relative.parts:raise ValueError('Task directory escapes the pinned asset root')
        directory=root/relative
        if not directory.resolve().is_relative_to(root):
            raise ValueError('Task directory escapes the pinned asset root')
        if not directory.is_dir():raise FileNotFoundError('Required task directory is missing: '+str(relative))
        for path in directory.rglob('*'):
            if path.is_symlink():raise ValueError('Task-owned assets must be regular files or directories: '+str(path))
            if path.is_dir():continue
            if not path.is_file():raise ValueError('Task-owned asset is not a regular file: '+str(path))
            files.add(str(path.relative_to(root)))
    return files


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args()
    spec=json.loads(a.spec.read_text());output=Path(spec['output']);output.mkdir(mode=0o700,parents=True,exist_ok=True)
    raw_credential=os.environ.pop('RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION')
    def status(phase,**fields):
        row={'phase':phase,**fields};(output/'status.json').write_text(json.dumps(row)+'\n')
        print('PEAR_L40S_ASSETS '+json.dumps(row),flush=True)
    status('installing_dependency_reader')
    deps=Path('/tmp/rp_jhh_usd_reader')
    subprocess.run([sys.executable,'-m','pip','install','--quiet','--target',str(deps),
        'usd-core==25.11','boto3==1.44.5'],check=True,timeout=600)
    sys.path.insert(0,str(deps))
    from pxr import Sdf,UsdUtils
    root=Path(spec['assets_root']).resolve();release=spec['release']
    if sha(root/release['release_manifest'])!=release['release_manifest_sha256']:
        raise ValueError('Source native asset release changed')
    for name,row in release['patched_files'].items():
        if sha(root/name)!=row['sha256_after']:raise ValueError('Source native asset patch changed')
    files={release['release_manifest'],*release['patched_files'],*spec['roots'],
        *task_owned_files(root,spec['roots'])};unresolved=set()
    for i,name in enumerate(sorted(spec['roots'])):
        path=root/name
        if not path.is_file():raise FileNotFoundError('Required asset root is missing: '+name)
        if path.suffix.lower()in{'.usd','.usda','.usdc','.usdz'}:
            layers,assets,missing=UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
            for value in [*[str(x.realPath)for x in layers],*assets]:
                dependency=Path(value)
                if not dependency.is_absolute():dependency=path.parent/dependency
                dependency=dependency.resolve()
                if dependency.is_file():
                    try:files.add(str(dependency.relative_to(root)))
                    except ValueError:raise ValueError('Native dependency is outside the pinned asset root: '+str(dependency))
            unresolved.update(missing)
        status('resolving_dependencies',checked=i+1,total=len(spec['roots']),files=len(files))
    # MDL modules are provided by the unchanged native simulator image.
    unexpected=[v for v in unresolved if not str(v).lower().endswith('.mdl')]
    if unexpected:
        (output/'unresolved.json').write_text(json.dumps(unexpected,indent=2))
        raise ValueError('Native asset dependencies remain unresolved; see the private receipt')
    paths=[root/name for name in sorted(files)]
    with ThreadPoolExecutor(max_workers=4)as executor:digests=list(executor.map(sha,paths))
    manifest={'files':{name:{'bytes':path.stat().st_size,'sha256':digest}
        for name,path,digest in zip(sorted(files),paths,digests,strict=True)},
        'release_manifest_sha256':release['release_manifest_sha256'],
        'image_provided_mdl_modules':sorted(unresolved),'training_templates':spec['templates']}
    package=output/'package';package.mkdir(exist_ok=True)
    (package/'asset-closure.json').write_text(json.dumps(manifest,sort_keys=True)+'\n')
    archive=package/'asset-closure.tar'
    with tarfile.open(archive,'w',dereference=True)as stream:
        for name in sorted(files):stream.add(root/name,arcname=name,recursive=False)
    status('uploading_asset_closure',files=len(files),bytes=archive.stat().st_size)
    credential=Path('/dev/shm/rp_jhh_asset_credentials.json')
    fd=os.open(credential,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w')as stream:stream.write(raw_credential)
    os.environ['PEAR_OBJECTSTORE_CREDENTIAL_FILE']=str(credential)
    try:
        from efe_pear_object_archive import archive as upload
        result=upload(package,spec['destination'],workers=2)
        if sha(root/release['release_manifest'])!=release['release_manifest_sha256']:
            raise ValueError('Source release changed while copying')
        status('asset_closure_archived',files=len(files),bytes=result['bytes'],
            uri=spec['destination'],readback_verified=result['readback_verified'])
    finally:credential.unlink(missing_ok=True)


if __name__=='__main__':main()
