"""Recover the interrupted ExIt save, preserving its failed attempt off quota."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path('/mnt/amlfs-03/home/rpunamiya/leact/studies/builderbench-gpt6-student-20261024-v1')
DEST = Path('/mnt/amlfs-07/shared/egoverse/rpunamiya/leact/studies/builderbench-gpt6-student-20261024-v1')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    from reproductions.leact.data import write_json
    from reproductions.leact.builderbench.iterate import select_and_export
    from reproductions.leact.builderbench.run_student import train_stage, phase
    from transformers import AutoTokenizer

    stage = ROOT / 'training/exit-r1'
    source = stage / 'sft'
    archive = DEST / 'failed-exit-save-20261024'
    fresh = DEST / 'training/exit-r1/sft'
    receipt = DEST / 'recovery-20261025.json'
    DEST.mkdir(parents=True, exist_ok=True)
    if not source.is_symlink():
        assert not (source / 'training_complete.json').exists(), 'Completed stage must not be retrained'
        assert not list((ROOT / 'ready').glob('exit-r1-*.json')), 'Published ExIt checkpoints must not be replaced'
        archive.mkdir(parents=True, exist_ok=True)
        records = []
        # Preserve and hash every file before releasing that file's original blocks.
        # A symlink leaves the failed attempt readable even if this operation is interrupted.
        for path in sorted(source.rglob('*')):
            if not path.is_file():
                continue
            target = archive / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                assert path.resolve() == target.resolve()
                continue
            digest = sha(path)
            if not target.exists() or sha(target) != digest:
                temporary = target.with_name(target.name + '.copying')
                with path.open('rb') as src, temporary.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
                    dst.flush(); os.fsync(dst.fileno())
                assert sha(temporary) == digest
                temporary.replace(target)
            assert sha(target) == digest
            records.append(dict(path=str(path.relative_to(source)), sha256=digest, bytes=path.stat().st_size))
            link = path.with_name(path.name + '.preserved-link')
            link.symlink_to(target)
            os.replace(link, path)
            print('PRESERVED', path.name, flush=True)
        fresh.mkdir(parents=True, exist_ok=True)
        source.rename(stage / 'sft.failed-quota-20261024')
        source.symlink_to(fresh, target_is_directory=True)
        write_json(receipt, dict(updated_utc=datetime.now(timezone.utc).isoformat(),
            reason='Disk quota exceeded during final epoch save after optimizer step 30',
            failed_attempt=str(archive), recovery='Restart ExIt only from unchanged joint-e3 anchor and unchanged selected data',
            destination=str(fresh), preserved=records))
    assert source.resolve() == fresh.resolve()
    write_json(ROOT / 'quota-recovery-ready.json', dict(storage=str(DEST), receipt=str(receipt)))
    anchor = Path(json.loads((ROOT / 'anchor-ready.json').read_text())['checkpoint'])
    original = json.loads((archive / 'manifest.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(anchor, local_files_only=True, trust_remote_code=False)
    manifest = select_and_export(ROOT / 'iteration-corpus', stage / 'generation', source,
                                 method='exit', tokenizer=tokenizer)
    assert manifest == original, 'Recovery must use identical ExIt training examples'
    train_stage(ROOT, 'exit-r1', source, anchor, method='exit', lr=1e-5, parent='joint-e3')
    write_json(ROOT / 'train-complete.json', dict(stages=sorted(p.stem for p in (ROOT / 'ready').glob('*.json')),
        next='Complete queued evaluations and fixed-observation timing', recovery_receipt=str(receipt)))
    phase(ROOT, 'train', 'complete', recovery_receipt=str(receipt))


if __name__ == '__main__':
    main()
