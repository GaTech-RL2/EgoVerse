"""Publish a completed local checkpoint with sequential, verified shared writes."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def archive(source,destination,*,reserve_bytes=80*1024**3):
    source=Path(source).resolve();destination=Path(destination)
    files={}
    for path in sorted(source.rglob('*')):
        if path.is_symlink():raise ValueError('A checkpoint archive cannot contain symbolic links')
        if path.is_file():files[str(path.relative_to(source))]={'bytes':path.stat().st_size}
    if not files:raise ValueError('Refusing to publish an empty checkpoint')
    if destination.exists():
        manifest=json.loads((destination/'rp_jhh_archive_manifest.json').read_text())
        if set(files)!=set(manifest['files']):raise ValueError('Checkpoint source differs from the published checkpoint')
        for name,value in manifest['files'].items():
            if (files[name]['bytes']!=value['bytes']or digest(source/name)!=value['sha256']or
                digest(destination/name)!=value['sha256']):
                raise ValueError('Previously published checkpoint differs')
        return manifest
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free<sum(v['bytes']for v in files.values())+reserve_bytes:
        raise RuntimeError('Insufficient shared space for a verified checkpoint plus the reserve')
    temporary=destination.with_name('.'+destination.name+'.partial-'+uuid4().hex)
    temporary.mkdir()
    try:
        for name,value in files.items():
            original=source/name;target=temporary/name;target.parent.mkdir(parents=True,exist_ok=True)
            before=original.stat();h=hashlib.sha256()
            with original.open('rb')as src,target.open('xb')as dst:
                for block in iter(lambda:src.read(8*1024**2),b''):dst.write(block);h.update(block)
                dst.flush();os.fsync(dst.fileno())
            after=original.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('Checkpoint changed during publication')
            if digest(target)!=h.hexdigest():raise ValueError('Checkpoint failed shared-storage readback')
            value['sha256']=h.hexdigest()
        manifest={'files':files,'bytes':sum(v['bytes']for v in files.values()),
                  'completed_utc':datetime.now(timezone.utc).isoformat(),'readback_verified':True}
        with (temporary/'rp_jhh_archive_manifest.json').open('x')as stream:
            stream.write(json.dumps(manifest,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise
    return manifest
