"""Archive native rollout grades and public media without changing the evaluator."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time


def now():return datetime.now(timezone.utc).isoformat()


def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.mirror.tmp');temporary.write_bytes(data);temporary.replace(path)


def mirror_once(source,destination,cache):
    receipts={};pending=[]
    names=['worker-start.json','worker-complete.json','progress.json','results.json','complete.json',
        'behavior.json','student-serving.json','student-preflight.json','student-serving-failure.json',
        'setup-error.json','student-prompt-manifest.json']
    for name in names:
        path=source/name
        if not path.is_file():continue
        data=path.read_bytes();json.loads(data)
        digest=hashlib.sha256(data).hexdigest()
        if cache.get(name)!=digest:atomic(destination/name,data);cache[name]=digest
        receipts[name]={'sha256':digest,'bytes':len(data)}
    media=source/'public-episodes/manifest.json'
    if media.is_file():
        data=media.read_bytes();manifest=json.loads(data)
        for row in manifest.get('episodes',[]):
            if Path(row['id']).name!=row['id']:raise ValueError('Unexpected public episode path')
            for name,key in [('trace.json','trace_sha256'),('replay.mp4','video_sha256')]:
                relative='public-episodes/'+row['id']+'/'+name;path=source/relative
                expected=row[key]
                if cache.get(relative)==expected:continue
                if not path.is_file():pending.append(relative);continue
                payload=path.read_bytes()
                if hashlib.sha256(payload).hexdigest()!=expected:pending.append(relative);continue
                atomic(destination/relative,payload);cache[relative]=expected
        if not pending:
            atomic(destination/'public-episodes/manifest.json',data)
            receipts['public-episodes/manifest.json']={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    prompt=source/'student-prompt-manifest.json'
    if prompt.is_file():
        for name,row in json.loads(prompt.read_text()).get('files',{}).items():
            if Path(name).name!=name:raise ValueError('Unexpected public prompt path')
            relative='student-prompt-requests/'+name;path=source/relative
            if cache.get(relative)==row['sha256']:continue
            if not path.is_file():pending.append(relative);continue
            data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=row['sha256']:pending.append(relative);continue
            atomic(destination/relative,data);cache[relative]=row['sha256']
    record={'updated_utc':now(),'source':str(source),'destination':str(destination),
        'receipt_files':receipts,'verified_files':dict(cache),'pending_files':pending,
        'scope':'Native grades and public artifacts only. Training, model calls, prompts, task membership and scoring are unchanged.'}
    atomic(destination/'mirror-receipt.json',(json.dumps(record,sort_keys=True)+'\n').encode())
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args()
    if not str(args.destination).startswith(('/mnt/amlfs-03/','/mnt/amlfs-01/home/rpunamiya/leact/')):raise ValueError('Expected shared project storage')
    cache={};deadline=time.monotonic()+40*3600
    while time.monotonic()<deadline:
        if not args.source.is_dir():
            time.sleep(10);continue
        try:
            record=mirror_once(args.source,args.destination,cache)
            if (args.source/'worker-complete.json').exists() and not record['pending_files']:
                atomic(args.destination/'mirror-complete.json',(json.dumps({'completed_utc':now(),'success':True})+'\n').encode())
                return
        except (OSError,ValueError,KeyError) as error:
            atomic(args.destination/'mirror-error.json',(json.dumps({'updated_utc':now(),'error':str(error)})+'\n').encode())
        time.sleep(2)


if __name__=='__main__':main()
