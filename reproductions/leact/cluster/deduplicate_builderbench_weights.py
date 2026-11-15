"""Share identical final-weight copies without changing bytes or checkpoint paths."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from pathlib import Path

ROOT=Path('/mnt/amlfs-03/home/rpunamiya/leact/studies/builderbench-gpt6-student-20261024-v1')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    jobs=[]
    for stage in ['joint','bc','leact-r1']:
        ready=json.loads((ROOT/'ready'/f'{stage}-e3.json').read_text())
        target=Path(ready['checkpoint'])
        for name,expected in ready['checkpoint_sha256'].items():
            source=target.parent/name
            assert source.is_file() and not source.is_symlink()
            jobs.append((source,target/name,expected))
    def verify(job):
        source,target,expected=job
        assert digest(source)==expected, str(source)
        assert digest(target)==expected, str(target)
        print('VERIFIED',source,flush=True)
        return dict(source=str(source),target=str(target),sha256=expected,bytes=source.stat().st_size)
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(verify,jobs))
    released=0
    for record in records:
        source,target=Path(record['source']),Path(record['target'])
        if source.stat().st_ino==target.stat().st_ino:continue
        link=source.with_name(source.name+'.deduplicate-link')
        link.unlink(missing_ok=True)
        os.link(target,link);os.replace(link,source)
        released+=record['bytes']
        assert source.stat().st_ino==target.stat().st_ino
    # The shared mount reports reclaimed blocks asynchronously.
    for _ in range(60):
        fs=os.statvfs(ROOT);available=fs.f_bavail*fs.f_frsize
        if available>=150_000_000_000:break
        time.sleep(5)
    record=dict(updated_utc=datetime.now(timezone.utc).isoformat(),released_bytes=released,
        available_bytes=available,method='Hard links after both copies match the published final checkpoint SHA256',files=records)
    (ROOT/'deduplication-20261025.json').write_text(json.dumps(record,indent=2)+'\n')
    assert available>=150_000_000_000, 'Need at least 150 GB before a fresh ExIt run'
    stage=ROOT/'training/exit-r1';source=stage/'sft'
    assert source.is_symlink(), 'Expected the failed first recovery on amlfs-07'
    assert not list((ROOT/'ready').glob('exit-r1-*.json'))
    source.rename(stage/'sft.failed-quota-20261025-retry')
    source.mkdir()
    (ROOT/'exit-recovery-storage-ready.json').write_text(json.dumps(record,indent=2)+'\n')
    print('STORAGE_READY',available,flush=True)


if __name__=='__main__':main()
