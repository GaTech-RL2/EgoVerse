"""Attach an I/O-only archive worker to an already allocated EFE node."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil

from .scale_efe import BASE,BUILD_ROOT,read,write
from .monitor_efe_preflight import remote


def ensure(workflow,archive,control):
    launch_file=control/'native-mirror-launch.json'
    previous=read(launch_file,{})
    if previous.get('workflow')==workflow:return previous
    source=BASE/'cluster/efe_native_artifact_mirror.py'
    expected=hashlib.sha256(source.read_bytes()).hexdigest()
    frozen=BUILD_ROOT/('efe-native-mirror-'+expected[:16]);frozen.mkdir(exist_ok=True)
    retained=frozen/source.name
    if not retained.exists():shutil.copyfile(source,retained)
    if hashlib.sha256(retained.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen archive worker differs')
    if not (frozen/'source-manifest.json').exists():write(frozen/'source-manifest.json',{source.name:expected})
    source=retained
    uploaded=subprocess.run(['osmo','workflow','rsync','upload',workflow,'preflight',str(source)+':/tmp',
        '--no-progress','--timeout','20'],text=True,capture_output=True,timeout=90)
    if uploaded.returncode:raise RuntimeError('Native artifact-mirror upload failed: '+uploaded.stderr[-1200:])
    destination=archive+'/native-artifacts/'+workflow
    script="""from pathlib import Path
import hashlib,json,subprocess,sys,os
source=Path('/tmp/efe_native_artifact_mirror.py')
if hashlib.sha256(source.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Archive worker source differs')
destination=Path(DESTINATION);destination.mkdir(parents=True,exist_ok=True)
receipt=destination/'mirror-launch.json'
old=json.loads(receipt.read_text()) if receipt.exists() else {}
pid=old.get('pid')
cmd=Path('/proc')/str(pid)/'cmdline'
running=pid and cmd.exists() and b'efe_native_artifact_mirror.py' in cmd.read_bytes()
if running:result=old
else:
 with (destination/'mirror.log').open('a') as log:
  child=subprocess.Popen([sys.executable,'-u',str(source),'--source',NATIVE_SOURCE,'--destination',str(destination)],
   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result={'pid':child.pid,'source_sha256':EXPECTED,'archive':str(destination),'workflow':WORKFLOW}
 receipt.write_text(json.dumps(result)+'\\n')
print('EFE_CAPTURE '+json.dumps(result),flush=True)
""".replace('EXPECTED',repr(expected)).replace('DESTINATION',repr(destination)).replace('NATIVE_SOURCE',repr('/tmp/efe-training-evaluation-v1/warm96/results/'+workflow)).replace('WORKFLOW',repr(workflow))
    receipt=remote(workflow,script);write(launch_file,receipt);return receipt
