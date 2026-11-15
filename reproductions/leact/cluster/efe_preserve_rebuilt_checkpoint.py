"""Preserve rebuilt weights and Adam state before releasing the failed handoff node."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

ROOT=Path('/tmp/efe-leact-warm96-rebuild-v2/leact-efe-warm96-rebuild-v4-1')
DEST=Path('/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-leact-rebuilt-20261030-v1/leact-efe-warm96-rebuild-v4-1')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)


def copy(source,target,expected=None):
    expected=expected or sha(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temporary=target.with_suffix(target.suffix+'.copy.tmp');shutil.copyfile(source,temporary)
        if sha(temporary)!=expected:raise ValueError('Artifact copy checksum differs: '+str(target))
        temporary.replace(target)
    if sha(target)!=expected:raise ValueError('Existing durable artifact differs: '+str(target))
    return {'sha256':expected,'bytes':target.stat().st_size}


def main():
    try:
        ready=json.loads((ROOT/'portable-ready.json').read_text())
        manifest=json.loads(Path(ready['manifest']).read_text())
        files={}
        for name,record in manifest['files'].items():
            files[name]=copy(Path(ready['checkpoint'])/name,DEST/'model'/name,record['sha256'])
            write(ROOT/'durable-copy-progress.json',{'phase':'weights','files':len(files),'total':len(manifest['files'])})
        copy(Path(ready['manifest']),DEST/'model/checkpoint-manifest.json',ready['manifest_sha256'])
        receipt={**ready,'checkpoint':str(DEST/'model'),'manifest':str(DEST/'model/checkpoint-manifest.json'),
            'all_files_verified':True,'completed_utc':datetime.now(timezone.utc).isoformat()}
        write(ROOT/'durable-model-ready.json',receipt);write(DEST/'model-ready.json',receipt)
        optimizer=copy(ROOT/'train/checkpoint-12/optimizer.pt',DEST/'training-state/optimizer.pt')
        for name in ['train/schedule.json','train/configuration.json','train/metrics.jsonl','train/complete.json',
            'reconstruction.json','weight-comparison.json','dataset-manifest.json','spec.json']:
            source=ROOT/name
            if source.is_file():copy(source,DEST/name)
        receipt.update(optimizer={'path':str(DEST/'training-state/optimizer.pt'),**optimizer},optimizer_persistence='Checksum-verified model and Adam state preserved on shared project storage.')
        write(ROOT/'durable-checkpoint-ready.json',receipt);write(DEST/'checkpoint-ready.json',receipt)
    except Exception as error:
        write(ROOT/'durable-copy-error.json',{'error':str(error),'updated_utc':datetime.now(timezone.utc).isoformat()});raise


if __name__=='__main__':main()
