"""Verify and relocate an inactive warmstart subtree, retaining its original data."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def inventory(root):
    value={}
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in sorted(dirs+files):
            p=Path(directory)/name;relative=str(p.relative_to(root))
            if p.is_symlink():value[relative]={'kind':'symlink','target':os.readlink(p)}
            elif p.is_file():value[relative]={'kind':'file','bytes':p.stat().st_size,'sha256':sha(p)}
            elif p.is_dir():value[relative]={'kind':'directory'}
            else:raise ValueError('Unsupported source file: '+str(p))
    return value


def relocate(source,destination,state,proof):
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(proof['utc'])).total_seconds()
    if not 0<=age<=120 or not proof.get('controller_stopped') or proof.get('active_warmstart_workflows'):
        raise ValueError('Fresh scheduler proof and stopped controller are required')
    source=source.resolve()
    if source.name!='warmstart' or source.is_symlink() or not source.is_dir():
        raise ValueError('Expected the current physical warmstart directory')
    destination.parent.mkdir(parents=True,exist_ok=True)
    state.mkdir(parents=True,exist_ok=True)
    staging=destination.with_name(destination.name+'.copying-20261113')
    backup=source.with_name(source.name+'.before-storage-relocation-20261113')
    redirect=source.with_name(source.name+'.redirect-20261113')
    for path in [destination,staging,backup,redirect]:
        if path.exists() or path.is_symlink():raise ValueError('Inspect existing relocation path: '+str(path))
    with (source/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(state/'progress.json',{'phase':'inventory','source':str(source),'destination':str(destination)})
        before=inventory(source)
        for name,row in before.items():
            if row['kind']=='symlink' and not Path(row['target']).is_absolute():
                target=(source/name).resolve()
                if target!=source and source not in target.parents:
                    raise ValueError('External relative symlink requires explicit preservation: '+name)
        write(state/'source-manifest.json',before)
        write(state/'progress.json',{'phase':'copying','entries':len(before),
            'bytes':sum(v.get('bytes',0) for v in before.values())})
        shutil.copytree(source,staging,symlinks=True)
        write(state/'progress.json',{'phase':'verifying','entries':len(before)})
        if inventory(source)!=before or inventory(staging)!=before:
            raise ValueError('Source changed or destination differs; source remains untouched')
        checkpoint=staging/'restart-states/checkpoint-176'
        receipt=json.loads((checkpoint/'training-state.json').read_text())
        if receipt['step']!=176 or receipt['target_step']!=737:raise ValueError('Unexpected committed restart')
        for name,row in receipt['files'].items():
            if Path(name).name!=name:raise ValueError('Nonlocal checkpoint member')
            p=checkpoint/name
            if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
                raise ValueError('Checkpoint receipt differs: '+name)
        staging.rename(destination)
        redirect.symlink_to(destination,target_is_directory=True)
        source.rename(backup)
        try:redirect.rename(source)
        except BaseException:
            backup.rename(source)
            raise
        if source.resolve()!=destination.resolve():raise ValueError('Redirect verification failed')
        result={'success':True,'utc':datetime.now(timezone.utc).isoformat(),
            'source':str(source),'destination':str(destination),'retained_source':str(backup),
            'verified_entries':len(before),'verified_bytes':sum(v.get('bytes',0) for v in before.values()),
            'source_manifest_sha256':sha(state/'source-manifest.json'),'checkpoint_step':176,
            'target_step':737,'recipe_changed':False,'original_data_deleted':False,'quiescence':proof}
        write(state/'complete.json',result)
        write(state/'progress.json',{'phase':'ready',**result})
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','destination','state','proof']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:relocate(args.source,args.destination,args.state,json.loads(args.proof.read_text()))
    except Exception as error:
        args.state.mkdir(parents=True,exist_ok=True)
        write(args.state/'error.json',{'error':repr(error),'utc':datetime.now(timezone.utc).isoformat()})
        raise
