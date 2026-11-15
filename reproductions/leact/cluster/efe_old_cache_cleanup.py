"""Inventory and remove the obsolete July download cache under explicit authorization.

Never follows links or touches checkpoints. The node-local inventory is the
deletion allowlist; every inode is checked again before unlinking.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import gzip
import json
import os
import stat

ROOT = Path('/mnt/amlfs-07/shared/rpunamiya')
WORK = Path('/tmp/efe-storage-cleanup-20261114')
CACHE = ROOT / 'dart/mecka16k'
CUTOFF = datetime(2026, 8, 16, tzinfo=timezone.utc).timestamp()


def write(name, value):
    p = WORK / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def fingerprint(p):
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_mtime >= CUTOFF or s.st_nlink != 1:
        raise ValueError('Not a singly-linked, old regular cache file: ' + str(p))
    return dict(path=str(p), inode=s.st_ino, bytes=s.st_size,
                allocated=s.st_blocks * 512, mtime_ns=s.st_mtime_ns)


def inventory():
    if CACHE.resolve() != CACHE or ROOT.resolve() != ROOT:
        raise ValueError('Cleanup root unexpectedly redirected')
    roots = [CACHE / 'data', CACHE / 'videos']
    tasks = []
    for root in roots:
        for p in root.iterdir():
            if p.is_symlink() or not p.is_dir() or not p.name.startswith('chunk-'):
                raise ValueError('Unexpected cache layout: ' + str(p))
            tasks.append(p)

    def scan(root):
        rows = []
        for parent, dirs, names in os.walk(root, followlinks=False):
            for name in dirs:
                q = Path(parent) / name
                if q.is_symlink():
                    raise ValueError('Unexpected symlink: ' + str(q))
            rows.extend(fingerprint(Path(parent) / name) for name in names)
        name = root.parent.name + '-' + root.name + '.jsonl.gz'
        with gzip.open(WORK / name, 'wt') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
        return dict(root=str(root), manifest=name, files=len(rows),
                    bytes=sum(r['bytes'] for r in rows), allocated=sum(r['allocated'] for r in rows))

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(scan, sorted(tasks)):
            results.append(row)
            if len(results) % 100 == 0:
                write('cache-audit-progress.json', dict(chunks=len(results), total=len(tasks)))
    core = ROOT / 'gr00t/core.15635'
    with core.open('rb') as f:
        header = f.read(20)
    if header[:4] != b'\x7fELF' or int.from_bytes(header[16:18], 'little') != 4:
        raise ValueError('Expected an ELF core dump')
    result = dict(status='audited', chunks=results, core=fingerprint(core),
                  utc=datetime.now(timezone.utc).isoformat(), cutoff=CUTOFF,
                  files=sum(r['files'] for r in results) + 1,
                  allocated=sum(r['allocated'] for r in results) + core.stat().st_blocks * 512,
                  retained=['dart/mecka16k/meta', 'dart/download_mecka16k.sh',
                            'dart/download_mecka16k.log', 'dart/dart_out', 'leact', 'all checkpoints'],
                  reason='Obsolete local download cache from killed July 23 job, plus April crash dump.')
    write('cache-audit-complete.json', result)


def apply():
    audit = json.loads((WORK / 'cache-audit-complete.json').read_text())
    approval = json.loads((WORK / 'cache-cleanup-authorization.json').read_text())
    if not approval.get('user_authorized_old_file_cleanup') or not approval.get('active_references_checked'):
        raise ValueError('Reviewed cleanup authorization is missing')
    if approval['audit_sha256'] != __import__('hashlib').sha256((WORK / 'cache-audit-complete.json').read_bytes()).hexdigest():
        raise ValueError('Reviewed inventory changed')
    removed = []

    def remove(row):
        p = Path(row['path'])
        if not (p.is_relative_to(CACHE / 'data') or p.is_relative_to(CACHE / 'videos')
                or p == ROOT / 'gr00t/core.15635'):
            raise ValueError('Outside cleanup allowlist')
        if fingerprint(p) != row:
            raise ValueError('File changed since audit: ' + str(p))
        p.unlink()

    def clear(chunk):
        with gzip.open(WORK / chunk['manifest'], 'rt') as f:
            for line in f:
                remove(json.loads(line))
        for parent, dirs, names in os.walk(chunk['root'], topdown=False, followlinks=False):
            Path(parent).rmdir()  # fails if a new file appeared
        return chunk

    with ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(clear, audit['chunks']):
            removed.append(row)
            if len(removed) % 50 == 0:
                v = os.statvfs(ROOT)
                write('cache-cleanup-progress.json', dict(chunks=len(removed), total=len(audit['chunks']),
                      removed_allocated=sum(r['allocated'] for r in removed), available_bytes=v.f_bavail*v.f_frsize))
    remove(audit['core'])
    v = os.statvfs(ROOT)
    write('cache-cleanup-complete.json', dict(status='completed', files=audit['files'],
          removed_allocated=audit['allocated'], available_bytes=v.f_bavail*v.f_frsize,
          retained=audit['retained'], utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inventory', 'apply'])
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    try:
        globals()[args.action]()
    except Exception as error:
        write('cache-' + args.action + '-error.json', {'error':repr(error)})
        raise
