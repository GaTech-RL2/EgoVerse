"""Execute the user's exact five-file optimizer cleanup with an audit receipt."""
from __future__ import annotations

from datetime import datetime,timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import stat


ROOT=Path('/mnt/amlfs-03/home/rpunamiya/leact/studies')
APPROVED={
    'efe-training-pilot-20261027-v5/leact-efe-training-pilot-v5-1/train/checkpoint-4/optimizer.pt',
    'efe-training-pilot-20261027-v5/leact-efe-training-pilot-v5-1/train/checkpoint-8/optimizer.pt',
    'efe-leact-pilot-20261027-v2/leact-efe-leact-pilot-v2-1/train/checkpoint-3/optimizer.pt',
    'efe-leact-pilot-20261027-v2/leact-efe-leact-pilot-v2-1/train/checkpoint-6/optimizer.pt',
    'efe-warm-bb-1e4-p3-20261027-v1-quota-retry4/leact-efe-warm-bb-1e4-p3-retry4-1/train/checkpoint-16/optimizer.pt',
}
AUDIT=ROOT/'efe-warm-continue32-96-20261028-v1/approved-storage-cleanup.json'


def fingerprint(path):
    s=path.stat()
    return {'bytes':s.st_size,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'links':s.st_nlink}


def write_receipt(value):
    AUDIT.parent.mkdir(parents=True,exist_ok=True)
    temporary=AUDIT.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True)+'\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(AUDIT)


def execute(proposal):
    if proposal.get('status')!='approved_by_user' or not proposal.get('approved_utc'):
        raise ValueError('Explicit user approval is required')
    requested={row['path'] for row in proposal['files']}
    if len(proposal['files'])!=5 or requested!={str(ROOT/name) for name in APPROVED}:
        raise ValueError('Cleanup differs from the approved five-file scope')
    signature=hashlib.sha256(json.dumps(proposal['files'],sort_keys=True).encode()).hexdigest()
    previous=json.loads(AUDIT.read_text()) if AUDIT.exists() else {}
    if previous and previous.get('approved_files_sha256')!=signature:
        raise ValueError('A different cleanup already owns this receipt')
    removed={row['path']:row for row in previous.get('removed',[])}
    # Validate ALL remaining candidates before the first deletion.
    for row in proposal['files']:
        path=Path(row['path'])
        if str(path) in removed:
            if path.exists():raise ValueError('Previously removed path reappeared')
            continue
        s=path.lstat()
        if not stat.S_ISREG(s.st_mode) or s.st_ino!=row['inode'] or s.st_size!=row['bytes'] or s.st_nlink!=row['links']:
            raise ValueError('Approved optimizer identity changed: '+str(path))
    protected={Path(p) for p in proposal['preserved_optimizer_files']}
    folders={Path(p).parent for p in requested}|{p.parent for p in protected}
    for folder in folders:
        manifest=json.loads((folder/'checkpoint-manifest.json').read_text())
        for name,row in manifest['files'].items():
            if Path(name).name!=name:raise ValueError('Unexpected manifest file path')
            path=folder/name
            if str(path) in requested:raise ValueError('Model manifest unexpectedly includes a deletion target')
            if path.stat().st_size!=row['bytes']:raise ValueError('Existing model file size differs from manifest')
            protected.add(path)
    before={str(p):fingerprint(p) for p in sorted(protected)}
    if previous.get('protected_before') and before!=previous['protected_before']:
        raise ValueError('A protected file changed since the cleanup began')
    volume=os.statvfs(ROOT)
    result=previous or {'status':'in_progress','approved_utc':proposal['approved_utc'],
        'approved_files_sha256':signature,'protected_before':before,'removed':[],
        'available_before_bytes':volume.f_bavail*volume.f_frsize}
    try:write_receipt(result)
    except OSError as exc:
        if exc.errno not in (errno.ENOSPC,errno.EDQUOT):raise
        # The approved operation frees the full shared volume. Retain its
        # preflight on node-local storage until the first unlink frees space.
        Path('/tmp/efe-approved-cleanup-preflight.json').write_text(json.dumps(result))
    for row in proposal['files']:
        path=Path(row['path'])
        if str(path) in removed:continue
        s=path.lstat()
        if s.st_ino!=row['inode'] or s.st_size!=row['bytes'] or not stat.S_ISREG(s.st_mode):
            raise ValueError('Optimizer changed after preflight')
        path.unlink()
        result['removed'].append(row)
        write_receipt(result)
    after={str(p):fingerprint(p) for p in sorted(protected)}
    if after!=before:raise ValueError('Protected model or retained optimizer metadata changed')
    volume=os.statvfs(ROOT)
    result.update(status='completed',removed_bytes=sum(r['bytes'] for r in result['removed']),
        protected_file_count=len(after),protected_metadata_unchanged=True,
        available_after_bytes=volume.f_bavail*volume.f_frsize,
        completed_utc=datetime.now(timezone.utc).isoformat())
    write_receipt(result)
    return result
