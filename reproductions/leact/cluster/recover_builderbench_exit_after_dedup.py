"""Restart only ExIt after verified duplicate removal and a capacity check."""
from pathlib import Path
import json
import os

ROOT=Path('/mnt/amlfs-03/home/rpunamiya/leact/studies/builderbench-gpt6-student-20261024-v1')


def main():
    from transformers import AutoTokenizer
    from reproductions.leact.data import write_json,file_sha256
    from reproductions.leact.builderbench.iterate import select_and_export
    from reproductions.leact.builderbench.run_student import train_stage,phase
    receipt=json.loads((ROOT/'exit-recovery-storage-ready.json').read_text())
    source=ROOT/'training/exit-r1/sft'
    assert not source.is_symlink()
    fs=os.statvfs(source)
    assert fs.f_bavail*fs.f_frsize>=150_000_000_000, 'Fresh ExIt requires 150 GB including retained epoch weights'
    anchor=Path(json.loads((ROOT/'anchor-ready.json').read_text())['checkpoint'])
    original=json.loads((source.parent/'sft.failed-quota-20261024/manifest.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(anchor,local_files_only=True,trust_remote_code=False)
    manifest=select_and_export(ROOT/'iteration-corpus',source.parent/'generation',source,method='exit',tokenizer=tokenizer)
    assert manifest==original, 'Recovery changed the selected training data'
    final=train_stage(ROOT,'exit-r1',source,anchor,method='exit',lr=1e-5,parent='joint-e3')
    # Preserve every epoch and the root path while avoiding another full copy.
    hashes=json.loads((ROOT/'ready/exit-r1-e3.json').read_text())['checkpoint_sha256']
    for name,expected in hashes.items():
        root_weight=final.parent/name;target=final/name
        assert file_sha256(root_weight)==expected and file_sha256(target)==expected
        link=root_weight.with_name(name+'.deduplicate-link')
        os.link(target,link);os.replace(link,root_weight)
    write_json(ROOT/'train-complete.json',dict(stages=sorted(p.stem for p in (ROOT/'ready').glob('*.json')),
        next='Complete queued evaluations and fixed-observation timing',
        recovery='ExIt restarted from the same warmstart with identical selected data after storage deduplication',
        deduplication_receipt=str(ROOT/'deduplication-20261025.json')))
    phase(ROOT,'train','complete')


if __name__=='__main__':main()
