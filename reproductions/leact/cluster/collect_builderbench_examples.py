"""Checksum-export selected saved traces for the offline comparison artifact."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import zlib
import pexpect

TASKS = ['cube-1-task-1', 'cube-4-task-2', 'cube-3-task-1',
         'cube-6-task-2', 'cube-2-task-2', 'cube-20-task-1']
STAGES = ['base', 'bc-e3', 'joint-e3', 'leact-r1-e3']
ROOT = '/mnt/amlfs-03/home/rpunamiya/leact/studies/builderbench-gpt6-student-20261024-v1'


def collect(workflow, output):
    script = f'''from pathlib import Path
import base64,hashlib,json,zlib
root=Path({ROOT!r})
paths=[]
for stage in {STAGES!r}:
 for task in {TASKS!r}:
  episode=root/'evaluations'/stage/'rollouts'/task/'seed-0'
  assert (episode/'result.json').exists(), str(episode)
  paths += [episode/'result.json', *sorted((episode/'calls').glob('*.json')), *sorted((episode/'steps').glob('*.json'))]
for method in ['leact','exit']:
 paths.append(root/'training'/f'{{method}}-r1'/'sft'/'selection.jsonl')
files=[]
for p in paths:
 data=p.read_bytes()
 files.append(dict(relative=str(p.relative_to(root)),sha256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode()))
raw=json.dumps(dict(root=str(root),files=files)).encode()
packed=base64.b64encode(zlib.compress(raw,9)).decode()
for i in range(0,len(packed),16000): print('TRACE_PART '+packed[i:i+16000],flush=True)
print('TRACE_DONE '+hashlib.sha256(raw).hexdigest(),flush=True)
'''
    packed=base64.b64encode(zlib.compress(script.encode(),9)).decode()
    entry='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
    child=pexpect.spawn('osmo',['workflow','exec',workflow,'train','--entry',entry],
                        encoding='utf-8',timeout=120,maxread=1_048_576,echo=False)
    chunks=[]
    try:
        while True:
            choice=child.expect([r'TRACE_PART ([A-Za-z0-9+/=]+)\r?\n',r'TRACE_DONE ([a-f0-9]{64})',pexpect.EOF],timeout=120)
            if choice==0: chunks.append(child.match.group(1))
            elif choice==1:
                digest=child.match.group(1);break
            else: raise RuntimeError(child.before[-3000:])
    finally:
        child.close(force=True)
    raw=zlib.decompress(base64.b64decode(''.join(chunks)))
    assert hashlib.sha256(raw).hexdigest()==digest
    bundle=json.loads(raw)
    output.mkdir(parents=True,exist_ok=True)
    evidence=[]
    for f in bundle['files']:
        relative=Path(f['relative'])
        assert not relative.is_absolute() and '..' not in relative.parts
        data=base64.b64decode(f['base64']);assert hashlib.sha256(data).hexdigest()==f['sha256']
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        evidence.append(dict(relative=str(relative),sha256=f['sha256'],bytes=len(data)))
    (output/'collection.json').write_text(json.dumps(dict(workflow=workflow,source=bundle['root'],
        bundle_sha256=digest,files=evidence),indent=2)+'\n')
    print(json.dumps(dict(files=len(evidence),bytes=sum(f['bytes'] for f in evidence),output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workflow');p.add_argument('output',type=Path)
    a=p.parse_args();collect(a.workflow,a.output)
